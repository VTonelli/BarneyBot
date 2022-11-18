from regex import D
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def sentence_preprocess(sentence, stopwords=stopwords.words(), min_sentence_len=3):
    """
    This function can be used in order to preprocess a given character sentence
    ## Params
    * `sentence`: the sentence to preprocess
    * `stopwords`: is a set of stopwords, the default value of such parameter is the set of 
    english stopwords of the library nltk
    * `min_sentence_len`: is the minimum lenght a sentence should have in order to be preprocessed
    and remain in the dataset
    ## Returns
    The preprocessed sentence
    """
    sentence = re.sub(r'[^A-Za-z\s]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence_splitted = sentence.split()

    is_short = len(sentence_splitted) <= min_sentence_len

    is_relevant = False
    for word in sentence_splitted:
        if word not in stopwords:
            is_relevant = True
            break

    return sentence, (not is_short and is_relevant)

class CTFIDFVectorizer(TfidfTransformer):
    """
    The goal of the class-based TF-IDF is to supply all documents within a single class
    with the same class vector. In order to do so, we have to start looking at TF-IDF 
    from a class-based point of view instead of individual documents.
    If documents are not individuals, but part of a larger collective, then it might be 
    interesting to actually regard them as such by joining all documents in a class together.
    """
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) 
        ## Params
        * `n_samples`: is the total number of unjoined documents.
        This is necessary as the IDF values become too small if the number of joined documents 
        is passed instead."""
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X

class FrequencyChatbotClassifier:
    """
    A word frequency classifier for characters classification
    """
    def __init__(self, characters, mode):
        """
        Construct a `FrequencyChatbotClassifier` 
        ## Params
        * `characters`: list of characters which corresponds to the labels of the model
        * `mode`: set the modality of the classifier based on the following possibilities:
           (`word frequency`: based on a count frecquency vectorizer, `tf-idf`: based on a 
            list of tf-idf, `c-tf-idf`: based on a class defined tf-idf)
        """
        self.characters = characters
        if mode != "word frequency" and mode != "tf-idf" and mode != "c-tf-idf":
            raise Exception("Unknown mode!")
        self.mode = mode
        self.loaded = False
        self.model = None
        self.distances = None

    def train(self, docs):
        """
        Train the model
        ## Params
        * `docs`: list of documents, in order to be done correctly the training, `len(docs)` must
        be equal to the number of characters you want to consider
        """

        assert len(docs) == len(self.characters)

        self.model = None
        self.loaded = False
        if len(docs) != len(self.characters):
            raise Exception(
                "Mismatch between classifier classes and provided documents!")
        
        # compute total length of the unjoined documents
        n_tot_docs = 0
        for doc_character in docs:
            n_tot_docs +=  len(doc_character)

        docs = [' '.join(doc) for doc in docs]
        if self.mode == 'word frequency':
            # fit count vectorizer
            vectorizer = CountVectorizer(stop_words='english').fit(docs)
            count = vectorizer.transform(docs)
            self.model = {
                'vectorizer': vectorizer,
                'vectors': count,
                'docs': docs
            }
        elif self.mode == 'tf-idf':
            tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')
            vectors = tfidf_vectorizer.fit_transform(docs)
            self.model = {
                'vectorizer': tfidf_vectorizer,
                'vectors': vectors,
                'docs': docs
            }
        elif self.mode == 'c-tf-idf':
            # fit count vectorizer
            count_vectorizer = CountVectorizer(stop_words='english').fit(docs)
            count = count_vectorizer.transform(docs)
            # fit c-tf-idf vectorizer
            ctfidf_vectorizer = CTFIDFVectorizer().fit(count, n_samples=n_tot_docs)
            ctfidf = ctfidf_vectorizer.transform(count)
            self.model = {
                'count_vectorizer': count_vectorizer,
                'vectorizer': ctfidf_vectorizer,
                'vectors': ctfidf,
                'docs': docs
            }
        self.loaded = True

    def predict(self, doc):
        """
        Classify a new document based on the technology previously selected in the `mode` parameter
        ## Params
        * `doc`: new document do classify
        ## Returns
        a dictionary where the keys are the character' names and their corresponding values are the 
        cosine similarity among the new document and the training documents
        """
        if not self.loaded:
            raise Exception("Classifier must be trained first!")
        doc1 = ' '.join(doc)
        predictions = dict()
        self.distances = None

        if self.mode == 'word frequency':
            test_vector = self.model['vectorizer'].transform([doc1])
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
        elif self.mode == 'tf-idf':
            test_vector = self.model['vectorizer'].transform([doc1])
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
        elif self.mode == 'c-tf-idf':
            count = self.model['count_vectorizer'].transform([doc1])
            test_vector = self.model['vectorizer'].transform(count)
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
            
        for character, idx_ch in zip(self.characters, range(len(self.characters))):
            predictions[character] = self.distances[idx_ch]
        
        return predictions

    def compute(self, doc, character):
        """
        Computes the wip score based on the classification, giving a proportional penalty in 
        case of misclassification
        ## Params
        * `doc`: document to evaluate
        * `character`: target character name 
        ## Returns
        a score in range [0, 1] of affinity personality for the document `doc` given the target 
        character named as specified in `character`, based on a IR techniques 
        """

        def zero_class(mu, pc):
            return 0 if mu == pc else mu
        
        predictions = self.predict(doc)
        values = np.array(list(predictions.values()))

        mu = np.max(values)
        pc = predictions[character]

        return pc * (1 - zero_class(mu, pc))