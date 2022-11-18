# All imports required by metrics in this library
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline, DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from operator import xor
import tensorflow as tf
import nlgmetricverse
from nlgmetricverse import NLGMetricverse
from torchmetrics import ExtendedEditDistance

from .metrics.triplet_nn_classifier import BarneyBotTripletClassifier
from .metrics.distinct import distinct
from .metrics.perplexity import perplexity
from .metrics.human import conversation, single_answers, consistency_questions
from .metrics.wmd import wmd
from .wip.wip import WIP

import evaluate

# Class defining a wrapper for any of the supported metrics, so that they can be loaded and computed seamlessly
class BBMetric:
    # List of supported metrics
    metrics_list = [
        "google bleu", "mpnet embedding similarity", "rouge l", "meteor", "emotion classifier",
        "roberta crossencoding similarity", "distinct", "neural chatbot classifier",
        "perplexity", "repetitiveness", "term error rate", "bertscore", "comet", "bleurt", "word mover distance", "bartscore",
        "extended edit distance", "wip"
    ]

    # Initialization
    def __init__(self, name, metric):
        self.metric = metric
        self.name = name
        self.compute_require_args = None
        self.train_require_args = None
        self.compute_optional_args = None
        self.train_optional_args = None
        self.return_args = None
        self.save_actors = None
        self.description = None
        self.paper = None
        # For each metric, define the required and optional parameters, as well as the dictionary entries returned
        if name == "google bleu":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "word mover distance":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "extended edit distance":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "bartscore":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score']
            self.save_actors = ['predictor', 'reference']
        elif name == "repetitiveness":
            self.compute_require_args = set(["sentences"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['document']
        elif name == "term error rate":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "meteor":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "rouge l":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "bleurt":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "mpnet embedding similarity":
            self.compute_require_args = set(["sentences_a", "sentences_b"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['document0', 'document1']
        elif name == "bertscore":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "emotion classifier":
            self.compute_require_args = set(["sentences"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['label', 'score', 'std']
            self.save_actors = ['document']
        elif name == "roberta crossencoding similarity":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference']
        elif name == "distinct":
            self.compute_require_args = set(["sentences"])
            self.compute_optional_args = set(["ngram_size"])
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['document']
        elif name == "neural chatbot classifier":
            self.compute_require_args = set(
                ["sentences", "character", "load_path"])
            self.compute_optional_args = set(["n_sentences", "verbose"])
            self.train_require_args = set([
                "character", "source_path", "source_encoded_path", "source_save_path",
                "random_state", "save_path"
            ])
            self.train_optional_args = set(
                ["shutdown_at_end", "n_shuffles"])
            self.return_args = ['score', 'std']
            self.save_actors = ['training_set', 'document']
        elif name == "perplexity":
            self.compute_require_args = set(["model", "encoded_test_set"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score']
            self.save_actors = ['training_set', 'predictor']
        elif name == "comet":
            self.compute_require_args = set(["sources", "predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            self.save_actors = ['predictor', 'reference', 'document']
        elif name == "wip":
            self.compute_require_args = set(["sentences", "character"])
            self.compute_optional_args = set([])
            self.train_require_args = set(["source_path", "mode"])
            self.train_optional_args = set()
            self.return_args = ['score']
            self.save_actors = ['training_set', 'document']

    # Pretty print metric
    def __str__(self):
        return str({
            "name": self.name,
            "args": {
                "train": {
                    "required": self.train_require_args,
                    "optional": self.train_optional_args
                },
                "compute": {
                    "required": self.compute_require_args,
                    "optional": self.compute_optional_args
                }
            },
            "returns": self.return_args,
            "description": self.description,
            "paper": self.paper,
            "save_actors": self.save_actors
        })

    # Function to load a metric, given its name
    @staticmethod
    def load_metric(name, **kwargs):
        metric = None
        # For each metric, based on the name, we instantiate either a function (algorithmic) or a transformer (neural)
        if name == "google bleu":
            metric = BBMetric(name, evaluate.load('google_bleu'))
        elif name == "repetitiveness":
            metric = BBMetric(name, NLGMetricverse(metrics=nlgmetricverse.load_metric("repetitiveness")))
        elif name == "bartscore":
            metric = BBMetric(name, NLGMetricverse(metrics=nlgmetricverse.load_metric("bartscore")))
        elif name == "term error rate":
            metric = BBMetric(name, evaluate.load('ter'))
        elif name == "word mover distance":
            metric = BBMetric(name, lambda a, b: wmd(a, b))
        elif name == "extended edit distance":
            metric = BBMetric(name, ExtendedEditDistance())
        elif name == "meteor":
            metric = BBMetric(name, evaluate.load('meteor'))
        elif name == "roberta crossencoding similarity":
            metric = BBMetric(name,
                              CrossEncoder('cross-encoder/stsb-roberta-large'))
        elif name == "rouge l":
            metric = BBMetric(name, evaluate.load('rouge'))
        elif name == "bleurt":
            metric = BBMetric(name, evaluate.load('bleurt', module_type="metric"))
        elif name == "bertscore":
            DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            DistilBertModel.from_pretrained("distilbert-base-uncased")
            metric = BBMetric(name, evaluate.load('bertscore'))
        elif name == "comet":
            metric = BBMetric(name, evaluate.load('comet', "eamt22-cometinho-da"))
        elif name == "emotion classifier":
            metric = BBMetric(
                name,
                pipeline(
                    "text-classification",
                    model='bhadresh-savani/distilbert-base-uncased-emotion',
                    return_all_scores=True))
        elif name == "perplexity":
            metric = BBMetric(name, lambda m, s: perplexity(m, s))
        elif name == "mpnet embedding similarity":
            metric = BBMetric(
                name,
                SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                ))
        elif name == "distinct":
            metric = BBMetric(name, lambda s, n: distinct(s, n))
        elif name == "neural chatbot classifier":
            metric = BBMetric(name, BarneyBotTripletClassifier())
        elif name == "wip":
            metric == BBMetric(name, WIP())
        else:
            raise Exception("Metric " + name + " is not supported!\n" +
                            "Supported metrics are " + str(BBMetric.metrics_list))
        return metric

    # Function to compute a metric on sentences/test set
    def compute(self, **kwargs):
        # Check that passed parameters are correct
        if not set(kwargs.keys()).issubset(
                set(self.compute_require_args).union(
                    set(self.compute_optional_args))):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.compute_require_args)
        if not set(self.compute_require_args).issubset(set(kwargs.keys())):
            raise Exception("Missing arguments! Required arguments are",
                            self.compute_require_args)
        # Dictionary in which to store results
        result = {}
        if self.name == "google bleu":
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            for i in range(len(predictions)):
                single_outputs.append(
                    self.metric.compute(predictions=[predictions[i]], references=[references[i]])['google_bleu'])
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "roberta crossencoding similarity":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            # Pass sentences to model, outputting similarity values
            single_outputs = self.metric.predict(
                list(zip(predictions, references)))
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "repetitiveness":
            # Cast predictions as list
            sentences = kwargs['sentences'] if type(
                kwargs['sentences']) is list else [kwargs['sentences']]
            single_outputs = []
            # Compute repetitiveness on single predictions
            for i in range(len(sentences)):
                single_outputs.append(self.metric(predictions=[sentences[i]],
                                                  references=[sentences[i]])['repetitiveness']['score'])
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "word mover distance":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            for i in range(len(predictions)):
                prediction = self.metric(predictions[i], references[i])
                if prediction is not None and not math.isinf(prediction):
                    single_outputs.append(prediction)
            # Write mean and std of these scores (if any)
            if single_outputs:
                result['score'] = np.mean(np.array(single_outputs))
                result['std'] = np.std(np.array(single_outputs))
            else:
                result['score'] = float('inf')
                result['std'] = float('nan')              
        elif self.name == "extended edit distance":
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            for i in range(len(predictions)):
                single_outputs.append(self.metric(predictions[i],
                                                  references[i]).item())
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "bartscore":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            result['score'] = self.metric(predictions=predictions,
                                          references=references)['bartscore']['score']
        elif self.name == "rouge l":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            # Compute separately for each prediction-reference pair the Rouge-L metric. Get the average f-measure
            for i in range(len(predictions)):
                single_outputs.append(self.metric.compute(predictions=[predictions[i]],
                                                          references=[references[i]])['rougeL'])
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "bertscore":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = self.metric.compute(predictions=predictions, references=references,
                                                 model_type="distilbert-base-uncased")['f1']
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "bleurt":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = self.metric.compute(predictions=predictions, references=references)['scores']
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "comet":
            # Cast predictions and references as lists
            sources = kwargs['sources'] if type(
                kwargs['sources']) is list else [kwargs['sources']]
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = self.metric.compute(predictions=predictions, references=references,
                                                 sources=sources)['scores']
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "term error rate":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            # Compute separately for each prediction-reference pair the Rouge-L metric. Get the average f-measure
            for i in range(len(predictions)):
                single_outputs.append(
                    self.metric.compute(predictions=[predictions[i]],references=[references[i]])['score'])
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "meteor":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            # Compute separately for each prediction-reference pair the METEOR metric
            for i in range(len(predictions)):
                single_outputs.append(
                    self.metric.compute(predictions=[predictions[i]], references=[references[i]])['meteor'])
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "emotion classifier":
            # Cast sentences as a list
            sentences = kwargs['sentences'] if type(
                kwargs['sentences']) is list else [kwargs['sentences']]
            # Pass sentences through the metric, outputting scores for each emotion label
            output = self.metric(sentences)
            result['score'] = {}
            result['std'] = {}
            # Initialize lists of scores for each emotion
            for emotion_dict in output[0]:
                result['score'][emotion_dict['label']] = []
            # For each sentence...
            for elem in output:
                # Append scores to the scores list for each emotion
                for emotion_dict in elem:
                    result['score'][emotion_dict['label']].append(emotion_dict['score'])
            # For each emotion label...
            for emotion_dict in output[0]:
                # Append emotion as a separate entry in the result dictionary
                emotion = emotion_dict['label']
                # Transform lists of scores into single std and mean values
                result['std'][emotion] = np.std(
                    np.array(result['score'][emotion]))
                result['score'][emotion] = np.mean(
                    np.array(result['score'][emotion]))
            # Return a dictionary with lists for labels, avg scores and stds, in corresponding order
            result['label'] = list(result['score'].keys())
            result['score'] = list(result['score'].values())
            result['std'] = list(result['std'].values())
        elif self.name == "mpnet embedding similarity":
            # Cast sentences as lists
            sentences_a = kwargs['sentences_a'] if type(
                kwargs['sentences_a']) is list else [kwargs['sentences_a']]
            sentences_b = kwargs['sentences_b'] if type(
                kwargs['sentences_b']) is list else [kwargs['sentences_b']]
            # Pass sentences through the model separately, then apply cosine similarity between the two encoded vectors, and
            # get diagonal values only (comparison between sentences_a_i and sentences_b_i for all i)
            single_outputs = np.diagonal(
                cosine_similarity(self.metric.encode(sentences_a),
                                  self.metric.encode(sentences_b)))
            # Compute mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "distinct":
            # Cast sentences as a list
            sentences = kwargs['sentences'] if type(
                kwargs['sentences']) is list else [kwargs['sentences']]
            # Compute distinct, obtaining mean and std for the metric
            result['score'], result['std'] = self.metric(
                sentences,
                kwargs['ngram_size'] if 'ngram_size' in kwargs else 3)
        # Compute perplexity or human metrics by simply passing params
        elif self.name == "perplexity":
            result['score'] = self.metric(kwargs['model'],
                                          kwargs['encoded_test_set'])
        elif self.name == "neural chatbot classifier":
            # Cast sentences as a list
            sentences = kwargs['sentences'] if type(
                kwargs['sentences']) is list else [kwargs['sentences']]
            # Assert there are enough sentences for the semantic classifier to perform a single evaluation
            if len(sentences) < 3:
                raise Exception(
                    "Needs at least three sentences to run the classifier!")
            # Perform sanity check on 'n_sentences' parameter
            n_sentences = kwargs[
                'n_sentences'] if 'n_sentences' in kwargs else 'all'
            if n_sentences != 'all' and type(n_sentences) != int:
                raise Exception("Invalid type for n_sentences!")
            if type(n_sentences) == int and n_sentences < 0:
                raise Exception("Invalid number of sentences!")
            if type(n_sentences) == int and n_sentences > len(sentences):
                n_sentences = 'all'
            # Compute the semantic classifier metric, returning scores for each sentences triple
            outputs = self.metric.compute(
                sentences, kwargs['character'], kwargs['load_path'],
                n_sentences, kwargs['verbose'] if 'verbose' in kwargs else False)
            # Compute mean and std for these values
            result['score'] = np.mean(np.array(outputs))
            result['std'] = np.std(np.array(outputs))
        elif self.name == "wip":
            # Cast sentences as a list
            sentences = kwargs['sentences'] if type(
                kwargs['sentences']) is list else [kwargs['sentences']]
            # Compute the semantic classifier metric, returning scores for each sentences triple
            outputs = self.metric.compute(sentences, kwargs['character'])
            result['score'] = np.array(outputs)

        # Sanitize type for the values of the result dictionary, so that it can be serialized
        for key in result:
            try:
                result[key] = result[key].tolist()
            except:
                pass
        # Return the results
        return result

    # Function to train a metric, which may be required eg. by neural metrics
    def train(self, **kwargs):
        # Check that passed parameters are correct
        if not set(kwargs.keys()).issubset(
                set(self.train_require_args).union(
                    set(self.train_optional_args))):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.train_require_args)
        if not set(self.train_require_args).issubset(set(kwargs.keys())):
            raise Exception("Missing Arguments! Required arguments are",
                            self.train_require_args)

        # If the metric does not require training, simply return
        if self.name == "google bleu" or \
           self.name == "repetitiveness" or \
           self.name == "term error rate" or \
           self.name == "meteor" or \
           self.name == "mpnet embedding similarity" or \
           self.name == "rouge l" or \
           self.name == "roberta crossencoding similarity" or \
           self.name == "emotion classifier" or \
           self.name == "distinct" or \
           self.name == "bertscore" or \
           self.name == "comet" or \
           self.name == "bleurt" or \
           self.name == "perplexity" or \
           self.name == "word mover distance" or \
           self.name == "extended edit distance" or \
           self.name == "bartscore":
            return
        # Otherwise, train the given metric, simply passing the required params
        elif self.name == "neural chatbot classifier":
            if not xor(bool(kwargs['source_path']), bool(kwargs['source_encoded_path'])):
                raise Exception("Exactly one between a source or an encoded source must be provided!")
            if kwargs['source_path'] and not kwargs['source_save_path']:
                raise Exception("When a non-encoded source is provided, a source save folder must also be provided!")
            if kwargs['source_encoded_path'] and kwargs['source_save_path']:
                print("Warning! A source save path has been provided but is unnecessary, and will be ignored.")
            self.metric.train(
                kwargs['character'], kwargs['source_path'], kwargs['source_encoded_path'],
                kwargs['source_save_path'], kwargs['save_path'], 
                kwargs['random_state'], kwargs['n_shuffles'] if 'n_shuffles' in kwargs else 10,
                kwargs['shutdown_at_end'] if 'shutdown_at_end' in kwargs else False)
        elif self.name == "wip":
            if not kwargs['source_path']:
                raise Exception("Source data path must be provided!")
            self.metric.train(kwargs['source_path'], kwargs['mode'])