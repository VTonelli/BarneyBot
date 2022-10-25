import pandas as pd


def get_tfidfs(docs, characters, tfidf_vectorizer):
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            index=characters,
                            columns=tfidf_vectorizer.get_feature_names())

    tfidfs = dict()

    for character in characters:
        tfidf_char = tfidf_df.loc[character].to_dict()
        tfidfs[character] = dict(
            sorted(tfidf_char.items(), key=lambda k: k[1], reverse=True))

    return tfidfs