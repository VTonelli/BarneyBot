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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
    
consistency_questions = ["Who are you?",
                         "What is your name?",
                         "What is your job?",
                         "Where do you live?"]

class BarneyBotTripletClassifier:
    def __init__(self):
        # Training params
        self.batch_size = 16
        self.lr = 1e-6
        self.patience = 6
        self.regularizer_weight_r = 1e-4
        self.regularizer_weight_s = 1e-3
        self.dropout_rate = 0.2
        self.train_size = 0.85
        self.test_size = 0.10
        # Instance state
        self.sentence_transformer = None
        self.character = None
        self.classifier_model = None

    def reset_state(self):
        self.sentence_transformer = None
        self.character = None
        self.classifier_model = None
        
    def create_model(self, input_size):
        inputs = keras.Input(shape=input_size)
        x = layers.Dense(
            1024,
            activation='relu',
        )(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(
            1024,
            activation='relu',
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            512, 
            activation='relu',
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            256, 
            activation='relu',
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(self.regularizer_weight_r),
            bias_regularizer=regularizers.l2(self.regularizer_weight_r)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        out = layers.Dense(
            1, 
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.regularizer_weight_s),
            bias_regularizer=regularizers.l2(self.regularizer_weight_s)
        )(x)
        classifier_model = keras.Model(inputs, out)
        classifier_model.compile(
            loss = keras.losses.BinaryCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate = self.lr),
            metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Recall()]
        )
        return classifier_model
    
    def get_triplet_df(self, series_df, n_shuffles, random_state):
        # separate character from others
        series_df_1 = series_df[series_df['character']==1].copy()
        series_df_0 = series_df[series_df['character']==0].copy()
        df_rows = {'character':[], 'encoded_lines':[]}
        for i in range(n_shuffles):
            print("Running shuffle " + str(i) + "/" + str(n_shuffles))
            # shuffle dataset
            series_df_1 = series_df_1.sample(frac=1, random_state=random_state+i).reset_index(drop=True)
            series_df_0 = series_df_0.sample(n=len(series_df_1), random_state=random_state+i).reset_index(drop=True)
            for i in tqdm(range(2,len(series_df_1))):
                # character
                lines = list(series_df_1['encoded_line'][i-2:i+1])
                lines = np.concatenate(lines)
                df_rows['character'].append(1)
                df_rows['encoded_lines'].append(lines)
                # other
                lines = list(series_df_0['encoded_line'][i-2:i+1])
                lines = np.concatenate(lines)
                df_rows['character'].append(0)
                df_rows['encoded_lines'].append(lines)
        df = pd.DataFrame(data=df_rows)
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    def compute(self, sentences, character, character_dict, base_folder, n_draws):
        character_folder = os.path.join(base_folder, "Data", "Characters", character)
        checkpoint_folder = os.path.join(character_folder, character_dict[character]['classifier_name'])
        if not self.classifier_model or character != self.character:
            self.classifier_model = keras.models.load_model(checkpoint_folder)
            self.character = character
        if not self.sentence_transformer:
            self.sentence_transformer = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        print("Using classifier at " + checkpoint_folder)
        samples = [self.sentence_transformer.encode(line) for line in sentences]
        n_samples = len(samples)
        triplets = []
        for _ in range(n_draws):
            rnd_indices = random.sample(range(0, n_samples), 3)
            triplets.append([samples[rnd_indices[0]], samples[rnd_indices[1]], samples[rnd_indices[2]]])
        input_size = len(samples[0]) * 3
        inputs = np.array(triplets).reshape(n_draws, input_size)
        outputs = self.classifier_model(inputs)
        return outputs
    
    def train(self, character, character_dict, source_dict, random_state, base_folder,
              n_shuffles=10, shutdown_at_end=False, from_saved_embeddings=True):
        self.reset_state()
        source = character_dict[character]['source']
        source_folder = os.path.join(base_folder, "Data", "Sources", source)
        character_folder = os.path.join(base_folder, "Data", "Characters", character)
        model_path = os.path.join(character_folder, character_dict[character]['classifier_name'])
        series_df = pd.read_csv(os.path.join(source_folder, source_dict[source]['df_filename']))
        series_df['character'] = series_df['character'].apply(lambda x: 1 if x==character else 0)
        series_df = series_df[['character', 'line']]
        if not os.path.exists(os.path.join(character_folder, character_dict[character]['encoded_lines_filename'])):
            from_saved_embeddings = False
            print('Encoded lines not found, from_saved_embeddings set to False')
        if not from_saved_embeddings:
            self.sentence_transformer = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            series_df['encoded_line'] = [self.sentence_transformer.encode(line) for line in tqdm(series_df['line'])]
            series_df[['line', 'character']].to_csv(
                os.path.join(character_folder, character_dict[character]['classifier_df']), 
                index = False
            )
            np.save(
                os.path.join(character_folder, character_dict[character]['encoded_lines_filename']),
                series_df['encoded_line'].to_numpy()
            )
            series_df = pd.read_csv(
                os.path.join(character_folder, character_dict[character]['classifier_df']),
                dtype={'line': str,
                       'character': int
                }
            )
            print("Saved encoded lines at", os.path.join(character_folder, character_dict[character]['encoded_lines_filename']))
        series_df['encoded_line'] = np.load(
            os.path.join(character_folder, character_dict[character]['encoded_lines_filename']), 
                         allow_pickle=True
        )
        print("Loaded encoded lines from", os.path.join(character_folder, character_dict[character]['encoded_lines_filename']))
        series_train_df, series_test_df = train_test_split(series_df, test_size=self.test_size, random_state=random_state)
        series_train_df, series_val_df = train_test_split(series_train_df, test_size = 1-self.train_size-self.test_size,
                                                          random_state=random_state)
        shuffled_df = self.get_triplet_df(series_df, n_shuffles=n_shuffles, random_state=random_state)
        tot_len = len(shuffled_df)
        train_len = int(tot_len*self.train_size)
        test_len = int(tot_len*self.test_size)
        val_len = tot_len - train_len - test_len
        print('Loading training data...')
        X_train = np.array([[float(e) for e in s] for s in tqdm(shuffled_df['encoded_lines'][:train_len])])
        y_train = np.array([c for c in tqdm(shuffled_df['character'][:train_len])])
        print('Loading test data...')
        X_test = np.array([[float(e) for e in s] for s in tqdm(shuffled_df['encoded_lines'][:test_len])])
        y_test = np.array([c for c in tqdm(shuffled_df['character'][:test_len])])
        print('Loading validation data...')
        X_val = np.array([[float(e) for e in s] for s in tqdm(shuffled_df['encoded_lines'][:val_len])])
        y_val = np.array([c for c in tqdm(shuffled_df['character'][:val_len])])
        self.classifier_model = self.create_model(len(X_train[0],))
        earlystop_callback = callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            min_delta=0,
            patience=self.patience,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        # Fit classifier
        train_history = self.classifier_model.fit(
            X_train, 
            y_train,
            validation_data = (X_val, y_val),
            epochs= 1000,
            verbose = 1, 
            callbacks=[earlystop_callback],
            batch_size = self.batch_size
        )
        self.character = character
        # Display confusion matrix
        print('#'*25 + ' Model Test ' + '#'*25)
        fig, ax=plt.subplots(1,1,figsize=(5,5))
        y_pred = self.classifier_model.predict(X_test).round()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Others', character])
        disp.plot(ax=ax)
        plt.show()
        # Save classifier and history
        classifier_path = os.path.join(character_folder, character_dict[character]['classifier_name'])
        self.classifier_model.save(classifier_path)
        filename = character.lower() + '_training_history.json'
        output_string = json.dumps(train_history.history)
        with open(os.path.join(character_folder, filename), 'w') as file:
            file.write(output_string)
        if shutdown_at_end:
            os.system('shutdown /' + shutdown_at_end)

def distinct(sentences, ngram_size=3):
    scores = []
    for sentence in sentences:
        distinct_ngrams = set(ngrams(sentence.split(), ngram_size))
        scores.append(len(distinct_ngrams) / len(sentence))
    return np.mean(np.array(scores)), np.std(np.array(scores))

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
            chat_history_ids = model.generate(bot_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                              do_sample=True, top_p=0.92, top_k=50)
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
        return np.average(human_convo_df['score'].to_numpy() / 5), np.std(human_convo_df['score'].to_numpy() / 5)
    
def single_answers(model, tokenizer, filepath, train, questions):
    questions_history = []
    if train:
        for question in questions:
            print("Question: {}".format(question))
            # encode the new user input, add the eos_token and return a tensor
            question_input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='tf')
            # generated a response while limiting the current answer to 128 tokens,
            max_length = 128 + question_input_ids.shape[1]
            chat_history_ids = model.generate(question_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                              do_sample=True, top_p=0.92, top_k=50)
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
        return np.average(human_convo_df['score'].to_numpy() / 5), np.std(human_convo_df['score'].to_numpy() / 5)

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
        self.return_args = None
        
        if name == "bleu":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
        elif name == "semantic similarity":
            self.compute_require_args = set(["sentences_a", "sentences_b"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
        elif name == "rouge l":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
        elif name == "emotion":
            self.compute_require_args = set(["sentences"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['label', 'score', 'std']
        elif name == "semantic answer similarity":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
        elif name == "distinct":
            self.compute_require_args = set(["sentences"])
            self.compute_optional_args = set(["ngram_size"])
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
        elif name == "semantic classifier":
            self.compute_require_args = set(["sentences", "character", "character_dict", "base_folder"])
            self.compute_optional_args = set(["n_draws"])
            self.train_require_args = set(["character", "character_dict", "source_dict", "random_state", "base_folder"])
            self.train_optional_args = set(["from_saved_embeddings", "shutdown_at_end", "n_shuffles"])
            self.return_args = ['score', 'std']
        elif name == "perplexity":
            self.compute_require_args = set(["model", "tokenizer", "sentences"])
            self.compute_optional_args = set(["stride"])
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score_concat']
        elif name == "human - coherence":
            self.compute_require_args = set(["filepath"])
            self.compute_optional_args = set()
            self.train_require_args = set(["model", "tokenizer", "filepath"])
            self.train_optional_args = set(["length"])
            self.return_args = ['score', 'std']
        elif name == "human - consistency":
            self.compute_require_args = set(["filepath"])
            self.compute_optional_args = set()
            self.train_require_args = set(["model", "tokenizer", "filepath"])
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
        elif name == "human - style":
            self.compute_require_args = set(["filepath"])
            self.compute_optional_args = set()
            self.train_require_args = set(["model", "tokenizer", "filepath", "questions"])
            self.train_optional_args = set()
            self.return_args = ['score', 'std']
            
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
                                       return_all_scores=True))
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
                              BarneyBotTripletClassifier())
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
            single_outputs = []
            for i in range(len(predictions)): 
                self.metric.add(prediction=predictions[i],
                                reference=references[i])
                single_outputs.append(self.metric.compute()['bleu'])
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "semantic similarity":
            sentences_a = kwargs['sentences_a'] if type(kwargs['sentences_a']) is list else [kwargs['sentences_a']]
            sentences_b = kwargs['sentences_b'] if type(kwargs['sentences_b']) is list else [kwargs['sentences_b']]
            single_outputs = self.metric.predict(list(zip(sentences_a, sentences_b)))
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "rouge l":
            predictions = kwargs['predictions'] if type(kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            for i in range(len(predictions)): 
                self.metric.add(prediction=predictions[i],
                                reference=references[i])
                single_outputs.append(self.metric.compute()['rougeL'].mid.fmeasure)
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "emotion":
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            output = self.metric(sentences)
            result['score'] = {}
            result['std'] = {}
            for emotion_dict in output[0]:
                result['score'][emotion_dict['label']] = []
            emotions_n = len(result['score'])
            for elem in output:
                for emotion_dict in elem:
                    result['score'][emotion_dict['label']].append(emotion_dict['score'])
            for emotion_dict in output[0]:
                emotion = emotion_dict['label']
                result['std'][emotion] = np.std(np.array(result['score'][emotion]))
                result['score'][emotion] = np.mean(np.array(result['score'][emotion]))
            result['label'] = list(result['score'].keys())
            result['score'] = list(result['score'].values())
            result['std'] = list(result['std'].values())
        elif self.name == "semantic answer similarity":
            predictions = kwargs['predictions'] if type(kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(kwargs['references']) is list else [kwargs['references']]
            single_outputs = np.diagonal(cosine_similarity(self.metric.encode(predictions),
                                                           self.metric.encode(references)))
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "distinct":
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            result['score'], result['std'] = self.metric(sentences,
                                                         kwargs['ngram_size'] if 'ngram_size' in kwargs else 3)
        elif self.name == "perplexity":
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            result['score_concat'] = self.metric(kwargs['model'], kwargs['tokenizer'], sentences,
                                                 kwargs['stride'] if 'stride' in kwargs else 64)
        elif self.name == "human - coherence":
            result['score'], result['std'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "human - consistency":
            result['score'], result['std'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "human - style":
            result['score'], result['std'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "semantic classifier":
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            if len(sentences) < 3:
                raise Exception("Needs at least three sentences to run the classifier!")
            outputs = self.metric.compute(sentences, kwargs['character'], kwargs['character_dict'], kwargs['base_folder'],
                                          kwargs['n_draws'] if 'n_draws' in kwargs else (len(sentences)-2))
            result['score'] = np.mean(np.array(outputs))
            result['std'] = np.std(np.array(outputs))
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
            self.metric.train(kwargs['character'], kwargs['character_dict'], kwargs['source_dict'],
                              kwargs['random_state'], kwargs['base_folder'],
                              kwargs['n_shuffles'] if 'n_shuffles' in kwargs else 10,
                              kwargs['shutdown_at_end'] if 'shutdown_at_end' in kwargs else False,
                              kwargs['from_saved_embeddings'] if 'from_saved_embeddings' in kwargs else True)
        elif self.name == "human - coherence":
            self.metric(kwargs['model'], kwargs['tokenizer'], kwargs['filepath'], True, 
                        kwargs['length'] if 'length' in kwargs else 5)
        elif self.name == "human - consistency":
            self.metric(kwargs['model'], kwargs['tokenizer'], kwargs['filepath'], True,
                        consistency_questions)
        elif self.name == "human - style":
            self.metric(kwargs['model'], kwargs['tokenizer'], kwargs['filepath'], True, 
                        kwargs['questions'])