import datasets
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Bozza
def perplexity(model, tokenizer, sentences):
    encodings = tokenizer("\n\n".join(sentences), return_tensors="tf")
    max_length = model.config.n_positions
    print(encodings.input_ids.shape)
    stride = 512
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.shape[1], stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.shape[1])
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = np.array(input_ids)
        target_ids[:, :-trg_len] = -100
        target_ids = tf.convert_to_tensor(target_ids)
        outputs = model({"input_ids": input_ids,
                         "labels": target_ids})
        neg_log_likelihood = outputs[0] * trg_len
        nlls.append(neg_log_likelihood)
    return torch.exp(torch.stack(nlls).sum() / end_loc) # Fix this line!

class BBMetric:
    def __init__(self, name, metric):
        self.metrics_list = ["bleu", "semantic similarity", "rouge l",
                             "emotion", "semantic answer similarity",
                             "semantic classifier"]
        self.metric = metric
        self.name = name
        self.compute_require_args = None
        self.train_require_args = None
        
        if name == "bleu":
            self.compute_require_args = set(["predictions", "references"])
            self.train_require_args = set()
        elif name == "semantic similarity":
            self.compute_require_args = set(["sentences_a", "sentences_b"])
            self.train_require_args = set()
        elif name == "rouge l":
            self.compute_require_args = set(["predictions", "references"])
            self.train_require_args = set()
        elif name == "emotion":
            self.compute_require_args = set(["sentences"])
            self.train_require_args = set()
        elif name == "semantic answer similarity":
            self.compute_require_args = set(["predictions", "references"])
            self.train_require_args = set()
        elif name == "semantic classifier":
            self.compute_require_args = set(["sentences"])
            self.train_require_args = set(["dataset", "_training_args"])
        elif name == "perplexity":
            self.compute_require_args = set(["model", "tokenizer", "sentence"])
            self.train_require_args = set()
        
    def __str__(self):
        return str({"instance": self, "name": self.name, "metric": self.metric})
    
    def load_metric(name, **kwargs):
        metric = None
        if name == "bleu":
            metric = BBMetric(name,
                              datasets.load_metric('bleu'))
        elif name == "semantic similarity":
            metric = BBMetric(name,
                              CrossEncoder('cross-encoder/stsb-roberta-large'))
        elif name == "rouge l":
            metric = BBMetric(name,
                              datasets.load_metric('rouge'))
        elif name == "emotion":
            metric = BBMetric(name,
                              pipeline("text-classification",
                                       model='bhadresh-savani/distilbert-base-uncased-emotion',
                                       return_all_scores=False))
        elif name == "perplexity":
            metric = BBMetric(name,
                              lambda m: perplexity(m))
        elif name == "semantic answer similarity":
            metric = BBMetric(name,
                              SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
        elif name == "semantic answer similarity classifier":
            metric = BBMetric(name,
                              SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
        else:
            raise Exception("Metric " + name + " is not supported!\n" +
                            "Supported metrics are ")
        return metric

    def compute(self, **kwargs):
        if not set(kwargs.keys()).issubset(set(self.compute_require_args)):
            raise Exception("Missing arguments! Required arguments are",
                            self.compute_require_args)
        if not set(self.compute_require_args).issubset(set(kwargs.keys())):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.compute_require_args)           
        result = {}
        if self.name == "bleu":
            predictions = [x.split() for x in kwargs['predictions']] if type(kwargs['predictions']) is list else [kwargs['predictions'].split()]
            references = [[x.split()] for x in kwargs['references']] if type(kwargs['references']) is list else [[kwargs['references'].split()]]
            self.metric.add_batch(predictions=predictions,
                                  references=references)
            result['score'] = self.metric.compute()['bleu']
        elif self.name == "semantic similarity":
            sentences_a = kwargs['sentences_a'] if type(kwargs['sentences_a']) is list else [kwargs['sentences_a']]
            sentences_b = kwargs['sentences_b'] if type(kwargs['sentences_b']) is list else [kwargs['sentences_b']]
            result['scores'] = self.metric.predict(list(zip(sentences_a, sentences_b)))
        elif self.name == "rouge l":
            predictions = kwargs['predictions'] if type(kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(kwargs['references']) is list else [kwargs['references']]
            self.metric.add_batch(predictions=predictions,
                                  references=references)
            result['score'] = self.metric.compute()['rougeL'].mid.fmeasure
        elif self.name == "emotion":
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            output = self.metric(sentences)
            result['scores'] = []
            result['labels'] = []
            for elem in output:
                result['scores'].append(elem['score'])
                result['labels'].append(elem['label'])
            result['scores'] = np.array(result['scores'])
        elif self.name == "semantic answer similarity":
            predictions = kwargs['predictions'] if type(kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(kwargs['references']) is list else [kwargs['references']]
            result['scores'] = np.diagonal(cosine_similarity(self.metric.encode(predictions),
                                                             self.metric.encode(references)))  
        elif self.name == "perplexity":
            raise NotImplementedError("Still Working on it!")
        elif self.name == "semantic classifier":
            raise NotImplementedError("Still Working on it!")
        return result
    
    def train(self, **kwargs):
        if not set(kwargs.keys()).issubset(set(self.train_require_args)):
            raise Exception("Missing arguments! Required arguments are",
                            self.train_require_args)
        if not set(self.train_require_args).issubset(set(kwargs.keys())):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.train_require_args)
        
        if self.name == "bleu" or \
           self.name == "semantic similarity" or \
           self.name == "rouge l" or \
           self.name == "semantic answer similarity" or \
           self.name == "emotion":
            return
        elif self.name == "semantic classifier":
            raise NotImplementedError("Still Working on it!")