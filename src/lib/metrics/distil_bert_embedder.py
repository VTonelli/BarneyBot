import random
from multiprocessing import cpu_count
from os.path import exists
from typing import List, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sentence_transformers import models, SentencesDataset
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
from torch.nn import Identity
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.BBData import character_dict, random_state

characters_all = list(character_dict.keys())
if 'Default' in characters_all:
    characters_all.remove('Default')

random.seed(random_state)


class BarneyEmbedder:

    def __init__(self,
                 embedder_path: str = None,
                 from_pretrained: bool = False,
                 use_cuda: bool = False) -> None:
        self.batch_size: int = 64

        self.embedding_size: int = 32

        self.lr: float = 1e-4
        self.epochs: int = 10
        self.training_steps: int = 30
        self.margin: int = self.embedding_size * 10

        self.model = self.create_model(embedder_path=embedder_path,
                                       from_pretrained=from_pretrained)
        if use_cuda:
            assert torch.cuda.is_available()
            device = torch.device('cuda')  # pylint: disable = no-member
            self.model.to(device)

    #

    def create_model(self,
                     embedder_path: str = None,
                     from_pretrained: bool = False) -> SentenceTransformer:
        if not from_pretrained:
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            model.train()
            dense = models.Dense(
                in_features=model.get_sentence_embedding_dimension(),
                out_features=self.embedding_size,
                activation_function=Identity())
            model.add_module('dense', dense)
        else:
            assert embedder_path is not None, 'Give model path to start from pre-trained model'
            assert exists(embedder_path
                          ), f'Current path "{embedder_path}" does not exists'
            model = SentenceTransformer(embedder_path)

        return model

    #

    def semi_hard_negative_mining(
            self,
            examples: List[InputExample],
            model: SentenceTransformer,
            margin: int,
            verbose: bool = False,
            parallel: bool = False) -> List[InputExample]:

        def _check_triplet(
                triplet_embedded: NDArray) -> Tuple[bool, bool, bool]:
            """
            returns
                - is triplet semi-hard
                - is triplet hard
                - is triplet easy
            """

            anchor_emb = triplet_embedded[0]
            positive_emb = triplet_embedded[1]
            negative_emb = triplet_embedded[2]

            dist_ap = np.linalg.norm(anchor_emb - positive_emb)
            dist_an = np.linalg.norm(anchor_emb - negative_emb)

            if dist_ap < dist_an:
                if dist_an < dist_ap + margin:
                    return True, False, False
                else:
                    return False, False, True
            else:
                return False, True, False

        ###############  create all embeddings  ###############
        if verbose:
            print('Creating all updated embeddings...')
        concat_examples = []
        for input_example in examples:
            concat_examples += input_example.texts

        ### concatenating triplets (anchor, positive, negative),
        ### the total lenght must be divisible by 3
        example_len = len(concat_examples)
        assert example_len % 3 == 0

        concat_embeddings = model.encode(concat_examples,
                                         show_progress_bar=verbose)
        embeddings = [[
            concat_embeddings[i], concat_embeddings[i + 1],
            concat_embeddings[i + 2]
        ] for i in range(0, example_len - 2, 3)]
        assert len(embeddings) == len(
            examples
        ), f'Lenght of embeddings ({len(embeddings)}) != lenght of examples ({len(examples)})'

        #

        ############### check examples hardness ###############
        if verbose:
            print('Checking triplets hardness...')
        if parallel:
            n_jobs = cpu_count()
            hardness_idxs = Parallel(n_jobs=n_jobs)(
                delayed(_check_triplet)(triplet)
                for triplet in tqdm(embeddings, disable=not verbose))
        else:
            hardness_idxs = [
                _check_triplet(triplet)
                for triplet in tqdm(embeddings, disable=not verbose)
            ]

        ### unpack indexes
        semi_hard_idxs = []
        hard_idxs = []
        easy_idxs = []
        for idx in hardness_idxs:
            semi_hard_idxs.append(idx[0])
            hard_idxs.append(idx[1])
            easy_idxs.append(idx[2])

        #

        ###############     filter dataset      ###############
        if verbose:
            print('Filtering dataset...')
        filtered = []
        for i, input_example in enumerate(examples):
            if semi_hard_idxs[i]:
                filtered.append(input_example)

        ###############    count statistics     ###############
        dataset_count = sum(semi_hard_idxs)
        hard_negatives_count = sum(hard_idxs)
        easy_positives_count = sum(easy_idxs)

        if verbose:
            print('#' * 30)
            print('Dataset length:      ', dataset_count)
            print('Hard negatives count:', hard_negatives_count)
            print('Easy positives count:', easy_positives_count)
            print('#' * 30, '\n')

        return filtered

    #

    def train(self,
              train_examples: List[InputExample],
              save_path: str,
              verbose: bool = False) -> None:

        ### set loss
        train_loss = TripletLoss(
            model=self.model,
            triplet_margin=self.margin,
            distance_metric=TripletDistanceMetric.EUCLIDEAN,
        )

        ### train loop
        for step in range(self.training_steps):
            if verbose:
                print('#' * 100)
                print(f'step {step+1}/{self.training_steps}')

            filtered_examples = self.semi_hard_negative_mining(train_examples,
                                                               self.model,
                                                               self.margin,
                                                               verbose=verbose)
            train_dataset = SentencesDataset(filtered_examples, self.model)
            train_dataloader = DataLoader(train_dataset,
                                          shuffle=True,
                                          batch_size=self.batch_size)

            if verbose:
                print('Training...')

            self.model.fit([(train_dataloader, train_loss)],
                           epochs=self.epochs,
                           optimizer_params={'lr': self.lr},
                           show_progress_bar=verbose)

        ### save model
        if save_path is None:
            print('Save path not setted, model will note be saved')
        else:
            self.model.save(save_path)

    #

    def compute(self, sentences: List[str], verbose: bool = False) -> NDArray:

        return self.model.encode(sentences, show_progress_bar=verbose)
