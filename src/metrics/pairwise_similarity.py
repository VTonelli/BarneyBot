from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
