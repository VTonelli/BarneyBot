import json
import random
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from numpy.typing import NDArray
from os import system
from tqdm import tqdm
from pandas import DataFrame, read_csv
from os.path import join, exists
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .distil_bert_embedder import BarneyEmbedder  # pylint: disable = relative-beyond-top-level
from sentence_transformers.readers import InputExample

from lib.BBData import character_dict, random_state

characters_all = list(character_dict.keys())
if 'Default' in characters_all:
    characters_all.remove('Default')

# random.seed(random_state)


class DistilBertClassifier:

    def __init__(self,
                 embedder_path: str = None,
                 from_pretrained: bool = False,
                 embedding_size: int = 32,
                 use_cuda: bool = False) -> None:
        self.characters = characters_all

        self.train_size = 0.85
        self.val_size = 0.05

        self.n_triplets_x_anchor: int = 10

        self.embedder = BarneyEmbedder(embedding_size=embedding_size,
                                       embedder_path=embedder_path,
                                       from_pretrained=from_pretrained,
                                       use_cuda=use_cuda)
        self.classifier = KNeighborsClassifier()

    #

    def set_characters(self, characters: List[str]) -> None:
        self.characters = characters

    def set_classifier(self, classifier) -> None:
        self.classifier = classifier

    #

    @staticmethod
    def get_character_df(series_df: DataFrame, n_shuffles: int,
                         n_sentences: int) -> DataFrame:
        # Separate lines by character from all the others
        series_df_char = series_df[series_df['character'] == 1].copy()
        # Define triplet dataset as having a character label and the line, already encoded
        df_rows = {'character': [], 'line': []}
        # Shuffle by a parametrized amount
        for i in range(n_shuffles):
            # print("Running shuffle " + str(i) + "/" + str(n_shuffles))
            # Shuffle the dataset and balance number of 0s (we suppose its cardinality is higher than that of 1s)
            series_df_char = series_df_char.sample(frac=1,
                                                   random_state=random_state +
                                                   i).reset_index(drop=True)
            # Iterate over lines
            for i in range(n_sentences, len(series_df_char) - n_sentences + 1):
                # Get a triple of consecutive lines for the character, and concatenate them in one sample
                lines = ' '.join(series_df_char['line'][i - n_sentences:i +
                                                        n_sentences])
                df_rows['character'].append(1)
                df_rows['line'].append(lines)
        # Create a new dataframe from the rows we have built
        df = DataFrame(data=df_rows)
        # Sample the dataset one last time to shuffle it
        return df.sample(frac=1,
                         random_state=random_state).reset_index(drop=True)

    def create_data(
            self,
            val: bool,
            source_encoded_path: str,
            n_shuffles: int = 2,
            merge_sentences: bool = True,
            n_sentences: int = 1,
            save_dataset: bool = True,
            verbose: bool = False) -> Tuple[List[DataFrame], List[DataFrame]]:

        # Flush the instance state cache
        # self.reset_state()

        if verbose:
            print('Loading encoded lines...')

        df_list = [
            read_csv(join(source_encoded_path, self.characters[c],
                          self.characters[c].lower() + '_classifier.csv'),
                     dtype={
                         'line': str,
                         'character': int
                     }) for c in range(len(self.characters))
        ]

        ### balance dataset
        max_len = min([len(df) for df in df_list])
        train_len = int(self.train_size * max_len)
        val_len = int(self.val_size * max_len) if val else 0
        val_end_idx = train_len + val_len
        df_list = [df[:max_len] for df in df_list]

        ### split in train and test
        df_list_train = []
        df_list_val = [] if val else None
        df_list_test = []
        for df in df_list:
            df_shuffled = df.sample(frac=1, random_state=random_state)

            df_list_train.append(df_shuffled[:train_len])
            df_list_test.append(df_shuffled[val_end_idx:])
            if val:
                df_list_val.append(df_shuffled[train_len:val_end_idx])

        ### augment dataset
        for c in tqdm(range(len(self.characters)), disable=not verbose):
            ### Load the preprocessed dataset
            series_df_train = df_list_train[c]
            series_df_test = df_list_test[c]
            if val: series_df_val = df_list_val[c]

            if merge_sentences:
                series_df_train = self.get_character_df(
                    series_df_train,
                    n_shuffles=n_shuffles,
                    n_sentences=n_sentences)
                series_df_test = self.get_character_df(series_df_test,
                                                       n_shuffles=n_shuffles,
                                                       n_sentences=n_sentences)
                if val:
                    series_df_val = self.get_character_df(
                        series_df_val,
                        n_shuffles=n_shuffles,
                        n_sentences=n_sentences)
            else:
                series_df_train = series_df_train[series_df_train['character']
                                                  == 1].reset_index()[[
                                                      'line', 'character'
                                                  ]]
                series_df_test = series_df_test[series_df_test['character'] ==
                                                1].reset_index()[[
                                                    'line', 'character'
                                                ]]
                if val:
                    series_df_val = series_df_val[series_df_val['character'] ==
                                                  1].reset_index()[[
                                                      'line', 'character'
                                                  ]]

            ### correct labels
            series_df_train['character'] = [
                c for _ in range(len(series_df_train))
            ]
            series_df_test['character'] = [
                c for _ in range(len(series_df_test))
            ]
            if val:
                series_df_val['character'] = [
                    c for _ in range(len(series_df_val))
                ]

            df_list_train[c] = series_df_train
            df_list_test[c] = series_df_test
            if val: df_list_val[c] = series_df_val

        ### save train and test datasets
        if save_dataset:
            with open(join(source_encoded_path, 'embedder_dataset_train.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump([df.to_dict() for df in df_list_train], file)

            with open(join(source_encoded_path, 'embedder_dataset_test.json'),
                      'w',
                      encoding='utf-8') as file:
                json.dump([df.to_dict() for df in df_list_test], file)

            if val:
                with open(join(source_encoded_path,
                               'embedder_dataset_val.json'),
                          'w',
                          encoding='utf-8') as file:
                    json.dump([df.to_dict() for df in df_list_val], file)

        return df_list_train, df_list_val, df_list_test

    def get_data(
        self,
        source_path: str,
        val: bool,
        override: bool = False,
        merge_sentences: bool = True,
        n_sentences: int = 2,
        verbose: bool = False,
    ) -> Tuple[List[str], List[int], List[str], List[int], List[str],
               List[int]]:

        ### create dataset if needed
        if override:
            df_list_train, df_list_val, df_list_test = self.create_data(
                val=val,
                source_encoded_path=source_path,
                merge_sentences=merge_sentences,
                n_sentences=n_sentences,
                verbose=verbose)
        else:
            with open(join(source_path, 'embedder_dataset_train.json'),
                      'r',
                      encoding='utf-8') as f:
                df_list_train = json.load(f)
            df_list_train = [DataFrame.from_dict(d) for d in df_list_train]

            if val:
                with open(join(source_path, 'embedder_dataset_val.json'),
                          'r',
                          encoding='utf-8') as f:
                    df_list_val = json.load(f)
                df_list_val = [DataFrame.from_dict(d) for d in df_list_val]

            with open(join(source_path, 'embedder_dataset_test.json'),
                      'r',
                      encoding='utf-8') as f:
                df_list_test = json.load(f)
            df_list_test = [DataFrame.from_dict(d) for d in df_list_test]

        X_train = sum([df['line'].tolist() for df in df_list_train], [])
        y_train = sum([df['character'].tolist() for df in df_list_train], [])
        X_val = sum([df['line'].tolist()
                     for df in df_list_val], []) if val else None
        y_val = sum([df['character'].tolist()
                     for df in df_list_val], []) if val else None
        X_test = sum([df['line'].tolist() for df in df_list_test], [])
        y_test = sum([df['character'].tolist() for df in df_list_test], [])

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_triplet_dataset(self,
                            X: List[str],
                            y: List[int],
                            verbose: bool = False) -> List[InputExample]:
        assert len(X) == len(y)

        #n_triplets_x_anchor = max(1, int(len(X) * n_triplets_x_anchor))
        if verbose:
            print('Creating triplets...')
        examples = []
        for i in tqdm(range(len(X)), disable=not verbose):
            y_ref = y[i]

            # pos_idxs = np.squeeze(np.where(y == y_ref))
            pos_idxs = [y_i for y_i in y if y_i == y_ref]
            random.shuffle(pos_idxs)
            # neg_idxs = np.squeeze(np.where(y != y_ref))
            neg_idxs = [y_i for y_i in y if y_i != y_ref]
            random.shuffle(neg_idxs)
            assert len(pos_idxs) > self.n_triplets_x_anchor
            assert len(neg_idxs) > self.n_triplets_x_anchor

            for pos in pos_idxs[:self.n_triplets_x_anchor]:
                for neg in neg_idxs[:self.n_triplets_x_anchor]:
                    positive = X[pos]
                    negative = X[neg]

                    examples.append(
                        InputExample(texts=[X[i], positive, negative]))

        random.shuffle(examples)

        return examples

    #

    def train_embedder(self,
                       patience: int,
                       train_examples: List[InputExample],
                       val_examples: List[InputExample],
                       save_path: str,
                       verbose: bool = False,
                       statistics_path: str = None) -> None:

        if verbose:
            print('Training embedder')

        return self.embedder.train(patience=patience,
                                   train_examples=train_examples,
                                   val_examples=val_examples,
                                   save_path=save_path,
                                   verbose=verbose,
                                   statistics_path=statistics_path)

    def train_classifier(self,
                         X_train: List[str],
                         y_train: List[int],
                         verbose: bool = False) -> None:
        if verbose:
            print('Training classifier')
        self.classifier = KNeighborsClassifier()
        train_embeddings = self.embedder.model.encode(
            X_train, show_progress_bar=verbose)
        self.classifier.fit(train_embeddings, y_train)

    def train(self,
              characters_path: str,
              model_path: str,
              train_embedder: bool = False,
              override_data: bool = False,
              merge_sentences: bool = True,
              n_sentences: int = 4,
              verbose: bool = False,
              test: bool = False,
              patience: int = None,
              save_statistics: bool = False,
              shutdown_at_end: str = None) -> None:

        ### if patience>0, create also validation set
        val = isinstance(patience, int) and patience >= 0

        ### if save statistics, define statistics folder
        statistics_path = join(model_path,
                               'statistics') if save_statistics else None

        ### get/create dataset
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data(
            source_path=characters_path,
            override=override_data,
            merge_sentences=merge_sentences,
            n_sentences=n_sentences,
            verbose=verbose,
            val=val,
        )

        if train_embedder:
            ### create triplet for triplet loss
            train_examples = self.get_triplet_dataset(X_train,
                                                      y_train,
                                                      verbose=verbose)
            ### create triplet for validation loss
            val_examples = self.get_triplet_dataset(
                X_val, y_val, verbose=verbose) if val else None

            ### train embedder
            self.train_embedder(patience=patience,
                                train_examples=train_examples,
                                val_examples=val_examples,
                                save_path=model_path,
                                verbose=verbose,
                                statistics_path=statistics_path)

        ### train classifier
        self.train_classifier(X_train=X_train,
                              y_train=y_train,
                              verbose=verbose)

        if test:
            self.test(X_test=X_test,
                      y_test=y_test,
                      verbose=verbose,
                      statistics_path=statistics_path)

        if shutdown_at_end is not None:
            if shutdown_at_end not in ['h', 's']:
                shutdown_at_end = 's'
            system('shutdown -' + shutdown_at_end)

    def test(self,
             X_test: List[str],
             y_test: List[int],
             verbose: bool = False,
             statistics_path: str = None) -> None:
        if verbose:
            print('Testing')

        embeddings = self.embedder.compute(sentences=X_test, verbose=verbose)
        predictions = self.classifier.predict(embeddings)  # maybe .ravel()
        assert len(self.characters) == len(
            set(y_test)
        ), f'Length of self.characters=={len(self.characters)} while y_test has {len(set(y_test))} different labels'

        cm = confusion_matrix(y_true=y_test,
                              y_pred=predictions,
                              labels=list(range(len(self.characters))),
                              normalize='pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.characters)
        disp.plot()
        plt.show()

        if statistics_path is not None:
            assert exists(statistics_path)
            disp.figure_.savefig(join(statistics_path, 'cm.png'))

    #

    def compute(self,
                sentences: List[str],
                verbose: bool = False,
                count_neighbors: bool = False) -> NDArray:
        embeddings = self.embedder.compute(sentences=sentences,
                                           verbose=verbose)
        if count_neighbors:
            predictions = self.classifier.kneighbors(
                embeddings, return_distance=False).ravel()
        else:
            predictions = self.classifier.predict(embeddings)

        predictions = np.array(list(collections.Counter(predictions).values()))
        return predictions / sum(predictions)
