from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

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

class FrequencyChatbotClassifier:
    def __init__(self, characters, mode):
        self.characters = characters
        if mode != "word frequency" and mode != "tf-idf":
            raise Exception("Unknown mode!")
        self.mode = mode
        self.loaded = False
        self.model = None

    def train(self, docs):
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
        self.loaded = True

    def predict(self, doc, mass=0.5):
        if not self.loaded:
            raise Exception("Classifier must be trained first!")
        doc = ' '.join(doc)
        predictions = dict()
        if self.mode == 'word frequency':
            v1 = filter_by_weights(get_word_frequency(doc), mass)
        elif self.mode == 'tf-idf':
            doc_names = self.characters.copy()
            doc_names.append('input')
            all_docs = self.model['docs'].copy()
            all_docs.append(doc)
            tfidfs = get_tfidfs(all_docs, doc_names, self.model['vectorizer'])
            v1 = filter_by_weights(tfidfs['input'], mass)
        for character in self.characters:
            if self.mode == 'word frequency':
                w = self.model[character]
            elif self.mode == 'tf-idf':
                w = tfidfs[character]
            v2 = filter_by_weights(w, mass)
            predictions[character] = freq_pairwise_sim(v1, v2)
        return predictions