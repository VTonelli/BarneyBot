import json
import random
import torch
from typing import List, Tuple
from numpy.typing import NDArray
from os import system
from tqdm import tqdm
from pandas import DataFrame, read_csv
from os.path import join
from sklearn.neighbors import KNeighborsClassifier

from .distil_bert_embedder import BarneyEmbedder
from sentence_transformers.readers import InputExample

from lib.BBData import character_dict, random_state

characters_all = list(character_dict.keys())
if 'Default' in characters_all:
    characters_all.remove('Default')

random.seed(random_state)


class DistilBertClassifier:

    def __init__(self,
                 embedder_path: str = None,
                 from_pretrained: bool = False,
                 use_cuda: bool = False) -> None:
        self.characters = characters_all

        self.train_size = 0.9

        self.n_triplets_x_anchor: int = 2
        self.n_sentences: int = 2

        self.embedder = BarneyEmbedder(embedder_path=embedder_path,
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
            source_encoded_path: str,
            n_shuffles: int = 10,
            merge_sentences: bool = True,
            n_sentences: int = 1,
            verbose: bool = False) -> Tuple[List[DataFrame], List[DataFrame]]:

        # Flush the instance state cache
        # self.reset_state()

        df_list = []
        if verbose:
            print('Loading encoded lines...')
        for c in tqdm(range(len(self.characters)), disable=not verbose):
            ### Load the preprocessed dataset
            series_df = read_csv(join(
                source_encoded_path, self.characters[c],
                self.characters[c].lower() + '_classifier.csv'),
                                 dtype={
                                     'line': str,
                                     'character': int
                                 })

            if merge_sentences:
                tmp_df = self.get_character_df(series_df,
                                               n_shuffles=n_shuffles,
                                               n_sentences=n_sentences)
            else:
                tmp_df = series_df[series_df['character'] == 1].reset_index()[[
                    'line', 'character'
                ]]
            tmp_df['character'] = [c for _ in range(len(tmp_df))]

            df_list.append(tmp_df)

        ### balance dataset
        max_len = min([len(df) for df in df_list])
        train_len = int(self.train_size * max_len)
        df_list = [df[:max_len] for df in df_list]

        ### split in train and test
        df_list_train = []
        df_list_test = []
        for df in df_list:
            df_shuffled = df.sample(frac=1, random_state=random_state)

            df_list_train.append(df_shuffled[:train_len])
            df_list_test.append(df_shuffled[train_len:])

        ### save train and test datasets
        with open(join(source_encoded_path, 'embedder_dataset_train.json'),
                  'w',
                  encoding='utf-8') as file:
            json.dump([df.to_dict() for df in df_list_train], file)

        with open(join(source_encoded_path, 'embedder_dataset_test.json'),
                  'w',
                  encoding='utf-8') as file:
            json.dump([df.to_dict() for df in df_list_test], file)

        return df_list_train, df_list_test

    def get_data(
        self,
        source_path: str,
        override: bool = False,
        merge_sentences: bool = True,
        n_sentences: int = 2,
        verbose: bool = False,
    ) -> Tuple[List[str], List[int], List[str], List[int]]:

        ### create dataset if needed
        if override:
            df_list_train, df_list_test = self.create_data(
                source_encoded_path=source_path,
                merge_sentences=merge_sentences,
                n_sentences=n_sentences,
                verbose=verbose)
        else:
            df_list_train = json.load(
                join(source_path, 'embedder_dataset_train.json'))
            df_list_train = [DataFrame.from_dict(d) for d in df_list_train]

            df_list_test = json.load(
                join(source_path, 'embedder_dataset_test.json'))
            df_list_train = [DataFrame.from_dict(d) for d in df_list_train]
        
        X_train = sum([df['line'].tolist() for df in df_list_train], [])
        y_train = sum([df['character'].tolist() for df in df_list_train], [])
        X_test = sum([df['line'].tolist() for df in df_list_test], [])
        y_test = sum([df['character'].tolist() for df in df_list_test], [])

        return X_train, y_train, X_test, y_test

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
                       train_examples: List[InputExample],
                       save_path: str,
                       verbose: bool = False) -> None:

        if verbose:
            print('Training embedder')

        return self.embedder.train(train_examples=train_examples,
                                   save_path=save_path,
                                   verbose=verbose)

    def train_classifier(self,
                         X_train: List[str],
                         y_train: List[int],
                         verbose: bool = False) -> None:
        if verbose:
            print('Training classifier')
        train_embeddings = self.embedder.model.encode(
            X_train, show_progress_bar=verbose)
        self.classifier.fit(train_embeddings, y_train)

    def train(self,
              characters_path: str,
              model_path: str,
              train_embedder: bool = False,
              override_data: bool = False,
              merge_sentences: bool = True,
              verbose: bool = False,
              shutdown_at_end: str = None) -> None:

        ### get/create dataset
        X_train, y_train, _, _ = self.get_data(
            source_path=characters_path,
            override=override_data,
            merge_sentences=merge_sentences,
            n_sentences=self.n_sentences,
            verbose=verbose,
        )

        if train_embedder:
            ### create triplet for triplet loss
            train_examples = self.get_triplet_dataset(X_train,
                                                      y_train,
                                                      verbose=verbose)

            ### train embedder
            self.train_embedder(train_examples=train_examples,
                                save_path=model_path,
                                verbose=verbose)

        ### train classifier
        self.train_classifier(X_train=X_train,
                              y_train=y_train,
                              verbose=verbose)

        if shutdown_at_end is not None:
            if shutdown_at_end not in ['h', 's']:
                shutdown_at_end = 's'
            system('shutdown -' + shutdown_at_end)

    #

    def compute(self, sentences: List[str], verbose: bool = False) -> NDArray:
        embeddings = self.embedder.compute(sentences=sentences,
                                           verbose=verbose)
        return self.classifier.predict(embeddings)
