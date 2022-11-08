from regex import D
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def sentence_preprocess(sentence, stopwords=stopwords.words(), min_sentence_len=3):
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
        `n_samples`: is the total number of unjoined documents.
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
            list of tf-idf, `c-tf-idf`: based on a class defined tf-idf 
        """
        self.characters = characters
        if mode != "word frequency" and mode != "tf-idf" and mode != "c-tf-idf":
            raise Exception("Unknown mode!")
        self.mode = mode
        self.loaded = False
        self.model = None
        self.distances = None

    def train(self, docs_per_class, n_tot_docs=None):
        """
        Train the model
        ## Params
        `docs_per_class`: list of document of for which the len is equal to the number of 
        characters
        """
        self.model = None
        self.loaded = False
        if len(docs_per_class) != len(self.characters):
            raise Exception(
                "Mismatch between classifier classes and provided documents!")
        docs_per_class = [' '.join(doc) for doc in docs_per_class]
        if self.mode == 'word frequency':
            # fit count vectorizer
            vectorizer = CountVectorizer(stop_words='english').fit(docs_per_class)
            count = vectorizer.transform(docs_per_class)
            self.model = {
                'vectorizer': vectorizer,
                'vectors': count,
                'docs_per_class': docs_per_class
            }
        elif self.mode == 'tf-idf':
            tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')
            vectors = tfidf_vectorizer.fit_transform(docs_per_class)
            self.model = {
                'vectorizer': tfidf_vectorizer,
                'vectors': vectors,
                'docs_per_class': docs_per_class
            }
        elif self.mode == 'c-tf-idf':
            # fit count vectorizer
            count_vectorizer = CountVectorizer(stop_words='english').fit(docs_per_class)
            count = count_vectorizer.transform(docs_per_class)
            # fit c-tf-idf vectorizer
            ctfidf_vectorizer = CTFIDFVectorizer().fit(count, n_samples=n_tot_docs)
            ctfidf = ctfidf_vectorizer.transform(count)
            self.model = {
                'count_vectorizer': count_vectorizer,
                'vectorizer': ctfidf_vectorizer,
                'vectors': ctfidf,
                'docs_per_class': docs_per_class
            }
        self.loaded = True

    def predict(self, doc, mass=0.5):
        if not self.loaded:
            raise Exception("Classifier must be trained first!")
        doc1 = ' '.join(doc)
        predictions = dict()
        self.distances = None
        if self.mode == 'word frequency':
            # v1 = filter_by_weights(get_word_frequency(doc1), mass)
            test_vector = self.model['vectorizer'].transform([doc1])
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
            # print(cosine_similarity(test_vector, self.model['vectors']))
        elif self.mode == 'tf-idf':
            test_vector = self.model['vectorizer'].transform([doc1])
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
            # return distances
            # doc_names = self.characters.copy()
            # doc_names.append('input')
            # all_docs = self.model['docs'].copy()
            # all_docs.append(doc1)
            # tfidfs = get_tfidfs(all_docs, doc_names, self.model['vectorizer'])
            # v1 = filter_by_weights(tfidfs['input'], mass)
        elif self.mode == 'c-tf-idf':
            count = self.model['count_vectorizer'].transform([doc1])
            test_vector = self.model['vectorizer'].transform(count)
            self.distances = cosine_similarity(test_vector, self.model['vectors'])[0]
            # return distances
            
        for character, idx_ch in zip(self.characters, range(len(self.characters))):
            # if self.mode == 'word frequency':
            #     w = self.model[character]
            # elif self.mode == 'tf-idf':
            #     w = tfidfs[character]
            # v2 = filter_by_weights(w, mass)
            # predictions[character] = freq_pairwise_sim(v1, v2)
            predictions[character] = self.distances[idx_ch]
        return predictions