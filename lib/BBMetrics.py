import datasets
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os
import pandas as pd
from nltk.util import ngrams

consistency_questions = ["Who are you?",
                         "What is your name?",
                         "What is your job?",
                         "Where do you live?"]

def distinct(sentences, ngram_size=3):
    scores = []
    for sentence in sentences:
        distinct_ngrams = set(ngrams(sentence.split(), ngram_size))
        scores.append(len(distinct_ngrams) / len(sentence))
    return np.mean(np.array(scores))

def perplexity(model, tokenizer, sentences, stride=64):
    encodings = tokenizer("\n\n".join(sentences), return_tensors="tf")
    max_length = model.config.n_positions
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
        neg_log_likelihood = sum(outputs['loss'].numpy())
        nlls.append(neg_log_likelihood)
    return np.exp(np.array(nlls).sum() / end_loc)

def human_conversation(model, tokenizer, filepath, train, length=5):
    chat_history = []
    if train:
        for step in range(length):
            # encode the new user input, add the eos_token and return a tensor
            user_sentence = input(">> User:")
            chat_history.append(user_sentence)
            new_user_input_ids = tokenizer.encode(user_sentence + tokenizer.eos_token, return_tensors='tf')
            # append the new user input tokens to the chat history
            bot_input_ids = tf.concat([chat_history_ids, new_user_input_ids], axis=-1) if step > 0 else new_user_input_ids
            # generated a response while limiting the current answer to 128 tokens,
            max_length = 128 + bot_input_ids.shape[1]
            chat_history_ids = model.generate(bot_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
            # pretty print last ouput tokens from bot
            bot_sentence = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            chat_history.append(bot_sentence)
            print("DialoGPT: {}".format(bot_sentence))
        got_score = False
        score = None
        while not got_score:
            score = input("How do you rate this conversation (0 to 5)? ")
            if score == "0" or score == "1" or score == "2" or score == "3" or score == "4" or score == "5":
                score = int(score)
                got_score = True
            else:
                print("Invalid score! Must be a single integer between 0 and 5!")
        if not os.path.exists(filepath.rsplit(os.path.sep, 1)[0]):
            os.makedirs(filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
        if os.path.exists(filepath):
            human_convo_df = pd.read_csv(filepath)
            human_convo_df = human_convo_df.append({"chat":chat_history, "score":score}, ignore_index=True)
        else:
            human_convo_df = pd.DataFrame.from_dict({"chat":[chat_history], "score":score})
        human_convo_df.to_csv(filepath, index=False)
    else:
        human_convo_df = pd.read_csv(filepath)
        average_score = np.average(human_convo_df['score'].to_numpy())
        return average_score / 5
    
def single_answers(model, tokenizer, filepath, train, questions):
    questions_history = []
    if train:
        for question in questions:
            print("Question: {}".format(question))
            # encode the new user input, add the eos_token and return a tensor
            question_input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='tf')
            # generated a response while limiting the current answer to 128 tokens,
            max_length = 128 + question_input_ids.shape[1]
            chat_history_ids = model.generate(question_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
            # pretty print last ouput tokens from bot
            bot_sentence = tokenizer.decode(chat_history_ids[:, question_input_ids.shape[-1]:][0], skip_special_tokens=True)
            questions_history.append((question, bot_sentence))
            print("DialoGPT: {}".format(bot_sentence))
        got_score = False
        score = None
        while not got_score:
            score = input("How do you rate these answers (0 to 5)? ")
            if score == "0" or score == "1" or score == "2" or score == "3" or score == "4" or score == "5":
                score = int(score)
                got_score = True
            else:
                print("Invalid score! Must be a single integer between 0 and 5!")
        if not os.path.exists(filepath.rsplit(os.path.sep, 1)[0]):
            os.makedirs(filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
        if os.path.exists(filepath):
            questions_df = pd.read_csv(filepath)
            questions_df = questions_df.append({"questions":questions_history, "score":score}, ignore_index=True)
        else:
            questions_df = pd.DataFrame.from_dict({"questions":[questions_history], "score":score})
        questions_df.to_csv(filepath, index=False)
    else:
        questions_df = pd.read_csv(filepath)
        average_score = np.average(questions_df['score'].to_numpy())
        return average_score / 5

class BBMetric:
    metrics_list = ["bleu", "semantic similarity", "rouge l",
                    "emotion", "semantic answer similarity", "distinct",
                    "semantic classifier", "perplexity", "human - coherence",
                    "human - consistency", "human - style"]
    def __init__(self, name, metric):
        self.metric = metric
        self.name = name
        self.compute_require_args = None
        self.train_require_args = None
        self.compute_optional_args = None
        self.train_optional_args = None
        
        if name == "bleu":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
        elif name == "semantic similarity":
            self.compute_require_args = set(["sentences_a", "sentences_b"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
        elif name == "rouge l":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
        elif name == "emotion":
            self.compute_require_args = set(["sentences"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
        elif name == "semantic answer similarity":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
        elif name == "distinct":
            self.compute_require_args = set(["sentences"])
            self.compute_optional_args = set(["ngram_size"])
            self.train_require_args = set()
            self.train_optional_args = set()
        elif name == "semantic classifier":
            self.compute_require_args = set(["sentences", "filepath"])
            self.compute_optional_args = set()
            self.train_require_args = set(["dataset", "character", "filepath"])
            self.train_optional_args = set()
        elif name == "perplexity":
            self.compute_require_args = set(["model", "tokenizer", "sentences"])
            self.compute_optional_args = set(["stride"])
            self.train_require_args = set()
            self.train_optional_args = set()
        elif name == "human - coherence":
            self.compute_require_args = set(["filepath"])
            self.compute_optional_args = set()
            self.train_require_args = set(["model", "tokenizer", "filepath"])
            self.train_optional_args = set(["length"])
        elif name == "human - consistency":
            self.compute_require_args = set(["filepath"])
            self.compute_optional_args = set()
            self.train_require_args = set(["model", "tokenizer", "filepath"])
            self.train_optional_args = set()
        elif name == "human - style":
            self.compute_require_args = set(["filepath"])
            self.compute_optional_args = set()
            self.train_require_args = set(["model", "tokenizer", "filepath", "questions"])
            self.train_optional_args = set()
            
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
                              lambda m, t, s, stride: perplexity(m, t, s, stride))
        elif name == "semantic answer similarity":
            metric = BBMetric(name,
                              SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
        elif name == "distinct":
            metric = BBMetric(name,
                              lambda s, n: distinct(s, n))
        elif name == "semantic classifier":
            metric = BBMetric(name,
                              SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
        elif name == "human - coherence":
            metric = BBMetric(name,
                              lambda m, t, f, train, l: human_conversation(m, t, f, train, l))
        elif name == "human - consistency":
            metric = BBMetric(name,
                              lambda m, t, f, train, q: single_answers(m, t, f, train, q))
        elif name == "human - style":
            metric = BBMetric(name,
                              lambda m, t, f, train, q: single_answers(m, t, f, train, q))
        else:
            raise Exception("Metric " + name + " is not supported!\n" +
                            "Supported metrics are ")
        return metric

    def compute(self, **kwargs):
        if not set(kwargs.keys()).issubset(set(self.compute_require_args).union(set(self.compute_optional_args))):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.compute_require_args)
        if not set(self.compute_require_args).issubset(set(kwargs.keys())):
            raise Exception("Missing arguments! Required arguments are",
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
        elif self.name == "distinct":
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            result['score'] = self.metric(sentences,
                                          kwargs['ngram_size'] if 'ngram_size' in kwargs else 3)
        elif self.name == "perplexity":
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            result['score'] = self.metric(kwargs['model'], kwargs['tokenizer'], sentences,
                                          kwargs['stride'] if 'stride' in kwargs else 64)
        elif self.name == "human - coherence":
            result['score'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "human - consistency":
            result['score'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "human - style":
            result['score'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "semantic classifier":
            raise NotImplementedError("Still Working on it!")
        return result
    
    def train(self, **kwargs):
        if not set(kwargs.keys()).issubset(set(self.train_require_args).union(set(self.train_optional_args))):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.train_require_args)
        if not set(self.train_require_args).issubset(set(kwargs.keys())):
            raise Exception("Missing Arguments! Required arguments are",
                            self.train_require_args)
        
        if self.name == "bleu" or \
           self.name == "semantic similarity" or \
           self.name == "rouge l" or \
           self.name == "semantic answer similarity" or \
           self.name == "emotion" or \
           self.name == "distinct" or \
           self.name == "perplexity":
            return
        elif self.name == "semantic classifier":
            raise NotImplementedError("Still Working on it!")
        elif self.name == "human - coherence":
            self.metric(kwargs['model'], kwargs['tokenizer'], kwargs['filepath'], True, 
                        kwargs['length'] if 'length' in kwargs else 5)
        elif self.name == "human - consistency":
            self.metric(kwargs['model'], kwargs['tokenizer'], kwargs['filepath'], True,
                        consistency_questions)
        elif self.name == "human - style":
            self.metric(kwargs['model'], kwargs['tokenizer'], kwargs['filepath'], True, 
                        kwargs['questions'])