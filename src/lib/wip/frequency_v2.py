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

nltk.download('stopwords')

def get_word_frequency(doc, f_sorted=False):
    wordlist = doc.lower().split()
    _temp = list()
    words = set(wordlist)
    wordfreq = {w: wordlist.count(w) for w in words - set(stopwords.words())}
    if f_sorted:
        wordfreq = dict(
            sorted(wordfreq.items(), key=lambda k: k[1], reverse=True))
    return wordfreq

def filter_by_weights(wordweights, mass):
    N = sum([v for v in wordweights.values()])
    n = 0
    filteredfreq = dict()
    for key, value in sorted(wordweights.items(),
                             key=lambda k: k[1],
                             reverse=True):
        n += value
        if n / N < mass:
            filteredfreq[key] = value
    return filteredfreq

def freq_pairwise_sim(v1, v2):
    wordset = set(v1.keys()).union((v2.keys()))
    v1_ord = list()
    v2_ord = list()
    for w in wordset:
        v1_ord.append(v1.get(w, 0.0))
        v2_ord.append(v2.get(w, 0.0))
    return cosine_similarity(
        np.array(v1_ord).reshape(1, -1),
        np.array(v2_ord).reshape(1, -1))[0][0]

def most_similar_words(v1, v2, max_words=15):
    wordlist = list(set(v1.keys()).intersection((v2.keys())))
    wordlist.sort(key=lambda word: abs(v1[word]-v2[word]))
    if max_words > len(wordlist):
        max_words = len(wordlist)
    return wordlist[:max_words]

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

def get_tfidfs(docs, characters, tfidf_vectorizer):
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            index=characters,
                            columns=tfidf_vectorizer.get_feature_names_out())

    tfidfs = dict()

    for character in characters:
        tfidf_char = tfidf_df.loc[character].to_dict()
        tfidfs[character] = dict(
            sorted(tfidf_char.items(), key=lambda k: k[1], reverse=True))

    return tfidfs

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

def get_ctfidfs(docs, characters, ctfidf_vectorizer, count_vectorizer):
    """Return a dictionary containing words and their corresponding c-tf-idf scores
    
    ## Parameters

    `docs`: collection of documents, it requires to be a  
    `charcters`: collection of class, namely the list of charcaters
    `ctfidf_vectorizer`: list of tfidf_vectorizers
    `count_vectorizer`: list of count_vectorizer
    """
    # tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    count = count_vectorizer.transform(docs)
    vector = ctfidf_vectorizer.transform(count)
    tfidf_df = pd.DataFrame(vector.toarray(),
                            index=characters,
                            columns=ctfidf_vectorizer.get_feature_names_out())

    tfidfs = dict()

    for character in characters:
        tfidf_char = tfidf_df.loc[character].to_dict()
        tfidfs[character] = dict(
            sorted(tfidf_char.items(), key=lambda k: k[1], reverse=True))

    return tfidfs

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

    def train(self, docs):
        """
        Train the model
        ## Params
        `docs`: list of document of for which the len is equal to the number of 
        characters
        """
        self.model = None
        self.loaded = False
        if len(docs) != len(self.characters):
            raise Exception(
                "Mismatch between classifier classes and provided documents!")
        docs = [' '.join(doc) for doc in docs]
        if self.mode == 'word frequency':
            self.model = dict()
            for i in range(len(self.characters)):
                self.model[self.characters[i]] = get_word_frequency(docs[i])
        elif self.mode == 'tf-idf':
            self.model = {
                'vectorizer':
                TfidfVectorizer(input='content', stop_words='english'),
                'docs':
                docs
            }
        elif self.mode == 'c-tf-idf':
            # fit count vectorizer
            count_vectorizer = CountVectorizer().fit(docs)
            count = count_vectorizer.transform(docs)
            # fit c-tf-idf vectorizer
            ctfidf_vectorizer = CTFIDFVectorizer().fit(count, n_samples=len(docs))
            ctfidf = ctfidf_vectorizer.transform(count)
            self.model = {
                'count_vectorizer': count_vectorizer,
                'vectorizer': ctfidf_vectorizer,
                'vectors': ctfidf,
                'docs': docs
            }
        self.loaded = True

    def predict(self, doc, mass=0.5):
        if not self.loaded:
            raise Exception("Classifier must be trained first!")
        doc1 = ' '.join(doc)
        predictions = dict()
        if self.mode == 'word frequency':
            v1 = filter_by_weights(get_word_frequency(doc1), mass)
        elif self.mode == 'tf-idf':
            doc_names = self.characters.copy()
            doc_names.append('input')
            all_docs = self.model['docs'].copy()
            all_docs.append(doc1)
            tfidfs = get_tfidfs(all_docs, doc_names, self.model['vectorizer'])
            v1 = filter_by_weights(tfidfs['input'], mass)
        elif self.mode == 'c-tf-idf':
            count = self.model['count_vectorizer'].transform([doc1])
            vector = self.model['vectorizer'].transform(count)
            distances = cosine_similarity(vector, self.model['vectors'])[0]
            return distances
            
        for character in self.characters:
            if self.mode == 'word frequency':
                w = self.model[character]
            elif self.mode == 'tf-idf':
                w = tfidfs[character]
            v2 = filter_by_weights(w, mass)
            predictions[character] = freq_pairwise_sim(v1, v2)
        return predictions

    def get_MSW(self, doc, mass=0.5, nwords=15):
        if not self.loaded:
            raise Exception("Classifier must be trained first!")
        doc1 = ' '.join(doc)
        msw_predictions = dict()
        if self.mode == 'word frequency':
            v1 = filter_by_weights(get_word_frequency(doc1), mass)
        elif self.mode == 'tf-idf':
            doc_names = self.characters.copy()
            doc_names.append('input')
            all_docs = self.model['docs'].copy()
            all_docs.append(doc1)
            tfidfs = get_tfidfs(all_docs, doc_names, self.model['vectorizer'])
            v1 = filter_by_weights(tfidfs['input'], mass)
        for character in self.characters:
            if self.mode == 'word frequency':
                w = self.model[character]
            elif self.mode == 'tf-idf':
                w = tfidfs[character]
            v2 = filter_by_weights(w, mass)
            msw_predictions[character] = most_similar_words(v1, v2, max_words=nwords)
        return msw_predictions