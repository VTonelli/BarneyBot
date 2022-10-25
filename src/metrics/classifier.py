from .frequency import get_word_frequency
from .filter import filter_by_weights
from .tfidf import get_tfidfs

from metrics import freq_pairwise_sim

from sklearn.feature_extraction.text import TfidfVectorizer


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