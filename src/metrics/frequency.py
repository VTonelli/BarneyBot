from nltk.corpus import stopwords


def get_word_frequency(doc, f_sorted=False):
    wordlist = doc.lower().split()
    _temp = list()
    words = set(wordlist)
    wordfreq = {w: wordlist.count(w) for w in words - set(stopwords.words())}
    if f_sorted:
        wordfreq = dict(
            sorted(wordfreq.items(), key=lambda k: k[1], reverse=True))
    return wordfreq
