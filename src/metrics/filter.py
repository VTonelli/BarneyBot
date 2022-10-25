import re


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


def sentence_filter_drop(sentence, min_sentence_len, relevant_words):
    sentence = re.sub(r'[^A-Za-z\s]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence_splitted = sentence.split()

    is_short = len(sentence_splitted) <= min_sentence_len

    is_relevant = False
    for word in sentence_splitted:
        if word in relevant_words:
            is_relevant = True
            break

    return is_short and not is_relevant
