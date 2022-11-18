# All imports required by metrics in this library
import datasets
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from operator import xor
from .metrics.triplet_nn_classifier import BarneyBotTripletClassifier
from .metrics.distinct import distinct
from .metrics.perplexity import perplexity
from .metrics.human import conversation, single_answers, consistency_questions

class WIPMetric:
    # List of supported metrics
    metrics_list = [
        "wip",
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

    # For each metric, define the required and optional parameters, as well as the dictionary entries returned
    if name == "wip":
        self.compute_require_args = set(["predictions", "references"])
        self.compute_optional_args = set()
        self.train_require_args = set("mode")
        self.train_optional_args = set()
        self.return_args = ['score', 'std']
    else:
        raise NotImplementedError()

    # Pretty print metric
    def __str__(self):
        return str({
            "instance": self,
            "name": self.name,
            "metric": self.metric
        })

    # Function to load a metric, given its name
    @staticmethod
    def load_metric(name, **kwargs):
        metric = None
        # For each metric, based on the name, we instantiate either a function (algorithmic) or a transformer (neural)
        if name == "bleu":
            metric = BBMetric(name, datasets.load_metric('bleu'))
        elif name == "semantic answer similarity":
            metric = BBMetric(name,
                              CrossEncoder('cross-encoder/stsb-roberta-large'))
        elif name == "rouge l":
            metric = BBMetric(name, datasets.load_metric('rouge'))
        elif name == "emotion":
            metric = BBMetric(
                name,
                pipeline(
                    "text-classification",
                    model='bhadresh-savani/distilbert-base-uncased-emotion',
                    return_all_scores=True))
        elif name == "perplexity":
            metric = BBMetric(name, lambda m, s: perplexity(m, s))
        elif name == "semantic similarity":
            metric = BBMetric(
                name,
                SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                ))
        elif name == "distinct":
            metric = BBMetric(name, lambda s, n: distinct(s, n))
        elif name == "neural chatbot classifier":
            metric = BBMetric(name, BarneyBotTripletClassifier())
        elif name == "human - coherence":
            metric = BBMetric(
                name, lambda m, t, f, train, l: conversation(
                    m, t, f, train, l))
        elif name == "human - consistency":
            metric = BBMetric(
                name,
                lambda m, t, f, train, q: single_answers(m, t, f, train, q))
        elif name == "human - style":
            metric = BBMetric(
                name,
                lambda m, t, f, train, q: single_answers(m, t, f, train, q))
        else:
            raise Exception("Metric " + name + " is not supported!\n" +
                            "Supported metrics are ")
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
        if self.name == "bleu":
            # Cast predictions and references as lists, separating tokens, as required by the HuggingFace BLEU metric
            predictions = [x.split() for x in kwargs['predictions']
                           ] if type(kwargs['predictions']) is list else [
                               kwargs['predictions'].split()
                           ]
            references = [[x.split()] for x in kwargs['references']
                          ] if type(kwargs['references']) is list else [[
                              kwargs['references'].split()
                          ]]
            # Compute via HuggingFace BLEU metric, and write the resulting score
            self.metric.add_batch(predictions=predictions,
                                  references=references)
            result['score'] = self.metric.compute()['bleu']
        elif self.name == "semantic answer similarity":
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
        elif self.name == "rouge l":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(
                kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(
                kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            # Compute separately for each prediction-reference pair the Rouge-L metric. Get the average f-measure
            for i in range(len(predictions)):
                self.metric.add(prediction=predictions[i],
                                reference=references[i])
                single_outputs.append(
                    self.metric.compute()['rougeL'].mid.fmeasure)
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "emotion":
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
                    result['score'][emotion_dict['label']].append(
                        emotion_dict['score'])
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
        elif self.name == "semantic similarity":
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
        elif self.name == "human - coherence":
            result['score'], result['std'] = self.metric(
                None, None, kwargs['filepath'], False, None)
        elif self.name == "human - consistency":
            result['score'], result['std'] = self.metric(
                None, None, kwargs['filepath'], False, None)
        elif self.name == "human - style":
            result['score'], result['std'] = self.metric(
                None, None, kwargs['filepath'], False, None)
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
        if self.name == "bleu" or \
           self.name == "semantic similarity" or \
           self.name == "rouge l" or \
           self.name == "semantic answer similarity" or \
           self.name == "emotion" or \
           self.name == "distinct" or \
           self.name == "perplexity":
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
        elif self.name == "human - coherence":
            self.metric(kwargs['model'], kwargs['tokenizer'],
                        kwargs['filepath'], True,
                        kwargs['length'] if 'length' in kwargs else 5)
        elif self.name == "human - consistency":
            self.metric(kwargs['model'], kwargs['tokenizer'],
                        kwargs['filepath'], True, consistency_questions)
        elif self.name == "human - style":
            self.metric(kwargs['model'], kwargs['tokenizer'],
                        kwargs['filepath'], True, kwargs['questions'])
