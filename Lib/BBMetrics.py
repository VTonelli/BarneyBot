# All imports required by metrics in this library
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
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import itertools

# Questions asked to all chatbots as part of the "human - consistency" metric
consistency_questions = ["Who are you?",
                         "What is your name?",
                         "What is your job?",
                         "Where do you live?"]

# Class defining the semantic classifier
class BarneyBotTripletClassifier:
    # Initialization
    def __init__(self):
        # Training params for the classifier
        self.batch_size = 16
        self.lr = 1e-6
        self.patience = 6
        self.regularizer_weight_r = 1e-4
        self.regularizer_weight_s = 1e-3
        self.dropout_rate = 0.2
        self.train_size = 0.85
        self.test_size = 0.10
        # Instance state, for caching, in case of repeated usage of this metric
        self.sentence_transformer = None
        self.character = None
        self.classifier_model = None
    # Function to flush instance state cache
    def reset_state(self):
        self.sentence_transformer = None
        self.character = None
        self.classifier_model = None
    # Function to create the keras model underneath the classifier
    def create_model(self, input_size):
        # Input is a concatenated triplet of sentences
        inputs = keras.Input(shape=input_size)
        # Model is a concatenation of dense layers alternated by batch normalizations
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
        # The last layers have L2 regularization, better suited for sigmoid output
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(self.regularizer_weight_r),
            bias_regularizer=regularizers.l2(self.regularizer_weight_r)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        # Output is a single probability value
        out = layers.Dense(
            1, 
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.regularizer_weight_s),
            bias_regularizer=regularizers.l2(self.regularizer_weight_s)
        )(x)
        # Create and compile keras model 
        classifier_model = keras.Model(inputs, out)
        classifier_model.compile(
            loss = keras.losses.BinaryCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate = self.lr),
            metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Recall()]
        )
        return classifier_model
    
    # Function to create a dataset composed of triples from a dataset of single sentences. Used in training only.
    def get_triplet_df(self, series_df, n_shuffles, random_state):
        # Separate lines by character from all the others
        series_df_1 = series_df[series_df['character']==1].copy()
        series_df_0 = series_df[series_df['character']==0].copy()
        # Define triplet dataset as having a character label and the line, already encoded
        df_rows = {'character':[], 'encoded_lines':[]}
        # Shuffle by a parametrized amount
        for i in range(n_shuffles):
            print("Running shuffle " + str(i) + "/" + str(n_shuffles))
            # Shuffle the dataset and balance number of 0s (we suppose its cardinality is higher than that of 1s)
            series_df_1 = series_df_1.sample(frac=1, random_state=random_state+i).reset_index(drop=True)
            series_df_0 = series_df_0.sample(n=len(series_df_1), random_state=random_state+i).reset_index(drop=True)
            # Iterate over lines
            for i in tqdm(range(2,len(series_df_1))):
                # Get a triple of consecutive lines for the character, and concatenate them in one sample
                lines = list(series_df_1['encoded_line'][i-2:i+1])
                lines = np.concatenate(lines)
                df_rows['character'].append(1)
                df_rows['encoded_lines'].append(lines)
                # Do the same for non-character lines
                lines = list(series_df_0['encoded_line'][i-2:i+1])
                lines = np.concatenate(lines)
                df_rows['character'].append(0)
                df_rows['encoded_lines'].append(lines)
        # Create a new dataframe from the rows we have built
        df = pd.DataFrame(data=df_rows)
        # Sample the dataset one last time to shuffle it
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Function to run the semantic classifier for a given character in evaluation mode
    def compute(self, sentences, character, character_dict, base_folder, n_sentences='all', verbose=False):
        # Get classifier checkpoint folder
        character_folder = os.path.join(base_folder, "Data", "Characters", character)
        checkpoint_folder = os.path.join(character_folder, character_dict[character]['classifier_name'])
        # If cached classifier is not the required one, re-load it
        if not self.classifier_model or character != self.character:
            self.classifier_model = keras.models.load_model(checkpoint_folder)
            self.character = character
        if not self.sentence_transformer:
            self.sentence_transformer = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        # Inform the user of successful loading
        if verbose:
            print("Using classifier at " + checkpoint_folder)
        # Encode single sentences
        samples = np.array([self.sentence_transformer.encode(line) for line in sentences])
        # Set a fixed random seed
        random.seed(1)
        # If n_sentences is set to 'all', we select all sentences. If it is an integer n, we instead randomly choose n sentences
        if type(n_sentences) == int:
            sampled_indices = np.random.randint(0, len(samples), size=n_sentences)
            samples = samples[sampled_indices]
        # Construct all triples from the selected sentences
        inputs = np.array([np.concatenate(triplet) for triplet in itertools.permutations(samples, 3)])
        # Get semantic classifier probability for each triple, and return all of them
        outputs = self.classifier_model(inputs)
        return outputs
    
    # Function to train the semantic classifier on a specific character
    def train(self, character, character_dict, source_dict, random_state, base_folder,
              n_shuffles=10, shutdown_at_end=False, from_saved_embeddings=True):
        # Flush the instance state cache
        self.reset_state()
        # Get the folder of the tv/series belonging to the character
        source = character_dict[character]['source']
        source_folder = os.path.join(base_folder, "Data", "Sources", source)
        # Get the character folder
        character_folder = os.path.join(base_folder, "Data", "Characters", character)
        # Get the path where the semantic classifier path will be saved
        model_path = os.path.join(character_folder, character_dict[character]['classifier_name'])
        # The (semantic classifier) encoded lines dataset is stored to speed up re-training if needed.
        # If it does not exists, then we create it now 
        if not os.path.exists(os.path.join(character_folder, character_dict[character]['encoded_lines_filename'])):
            from_saved_embeddings = False
            print('Encoded lines not found, from_saved_embeddings set to False')
        # Create the encoded lines dataset
        if not from_saved_embeddings:
            # Read the tv/series dataset of the character
            series_df = pd.read_csv(os.path.join(source_folder, source_dict[source]['df_filename']))
            # Apply class labelling to the dataset sentences
            series_df['character'] = series_df['character'].apply(lambda x: 1 if x==character else 0)
            # Throw away unnecessary dataset rows
            series_df = series_df[['character', 'line']]
            # Load the sentence transformer to encode lines
            self.sentence_transformer = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            # Encode lines and add them to the dataset as a new row
            series_df['encoded_line'] = [self.sentence_transformer.encode(line) for line in tqdm(series_df['line'])]
            # Save the dataset rows as a csv file
            series_df[['line', 'character']].to_csv(
                os.path.join(character_folder, character_dict[character]['classifier_df']), 
                index = False
            )
            # The encoded lines are saved separately via numpy due to their type (array)
            np.save(
                os.path.join(character_folder, character_dict[character]['encoded_lines_filename']),
                series_df['encoded_line'].to_numpy()
            )
            print("Saved encoded lines at", os.path.join(character_folder, character_dict[character]['encoded_lines_filename']))
        # Load the preprocessed dataset
        series_df = pd.read_csv(
                os.path.join(character_folder, character_dict[character]['classifier_df']),
                dtype={'line': str,
                       'character': int
                }
        )
        # Load encoded lines dataset, via numpy, and add it as a new row in the dataset
        series_df['encoded_line'] = np.load(
            os.path.join(character_folder, character_dict[character]['encoded_lines_filename']), 
                         allow_pickle=True
        )
        print("Loaded encoded lines from", os.path.join(character_folder, character_dict[character]['encoded_lines_filename']))
        # Perform train-val-test split on the dataset
        series_train_df, series_test_df = train_test_split(series_df, test_size=self.test_size, random_state=random_state)
        series_train_df, series_val_df = train_test_split(series_train_df, test_size = 1-self.train_size-self.test_size,
                                                          random_state=random_state)
        # Get triples from the dataset
        shuffled_df = self.get_triplet_df(series_df, n_shuffles=n_shuffles, random_state=random_state)
        # Store into variables the train, val, test, total lengths of the new (triplets) dataset
        tot_len = len(shuffled_df)
        train_len = int(tot_len*self.train_size)
        test_len = int(tot_len*self.test_size)
        val_len = tot_len - train_len - test_len
        # Load triples into numpy arrays, separating data and labels
        print('Loading training data...')
        X_train = np.array([[float(e) for e in s] for s in tqdm(shuffled_df['encoded_lines'][:train_len])])
        y_train = np.array([c for c in tqdm(shuffled_df['character'][:train_len])])
        print('Loading test data...')
        X_test = np.array([[float(e) for e in s] for s in tqdm(shuffled_df['encoded_lines'][:test_len])])
        y_test = np.array([c for c in tqdm(shuffled_df['character'][:test_len])])
        print('Loading validation data...')
        X_val = np.array([[float(e) for e in s] for s in tqdm(shuffled_df['encoded_lines'][:val_len])])
        y_val = np.array([c for c in tqdm(shuffled_df['character'][:val_len])])
        # Create the keras model for the semantic classifier
        self.classifier_model = self.create_model(len(X_train[0],))
        # Define early stop behavior
        earlystop_callback = callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            min_delta=0,
            patience=self.patience,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        # Fit the semantic classifier
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
        # Display a confusion matrix, to show the results of the semantic classifier
        print('#'*25 + ' Model Test ' + '#'*25)
        fig, ax=plt.subplots(1,1,figsize=(5,5))
        y_pred = self.classifier_model.predict(X_test).round()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Others', character])
        disp.plot(ax=ax)
        plt.show()
        # Save the semantic classifier and its training history
        classifier_path = os.path.join(character_folder, character_dict[character]['classifier_name'])
        self.classifier_model.save(classifier_path)
        filename = character.lower() + '_training_history.json'
        output_string = json.dumps(train_history.history)
        with open(os.path.join(character_folder, filename), 'w') as file:
            file.write(output_string)
        # If a shutdown at the end is required, do so
        if shutdown_at_end:
            os.system('shutdown /' + shutdown_at_end)

# Function to compute the distinct metric
def distinct(sentences, ngram_size=3):
    scores = []
    # For each sentence...
    for sentence in sentences:
        # Get the ngrams of required size, and encode them as a set (-> no repeated elements)
        distinct_ngrams = set(ngrams(sentence.split(), ngram_size))
        # Divide the length of this set by the number of tokens in the sentence (approx. = the number of ngrams) to get the distinct
        # score for this sentence
        scores.append(len(distinct_ngrams) / len(sentence-(ngram_size-1)))
    # Compute mean and std of scores
    return np.mean(np.array(scores)), np.std(np.array(scores))

# Function to compute the perplexity of the model
def perplexity(model, encoded_test_set):
    # Get the maximum allowed length by the model
    max_length = model.config.n_positions
    # list of negative log-likelihoods
    nlls = []
    # Get an iterator over the test set, already encoded by the tokenizer
    iterator = iter(encoded_test_set)
    # For each iteration...
    for _ in tqdm(range(len(encoded_test_set))):
        # Get the batch, and evaluate it through the model, returning the loss (= average nll)
        batch = next(iterator)
        loss = model.evaluate(batch, verbose=0)
        # Append the nll to the list
        nlls.append(loss)
    # Average over the nlls and exponentiate them to get the perplexity score
    return np.exp(np.array(nlls).sum() / len(encoded_test_set))

# Function to run a conversation between a chatbot and a human, used to train the "human - coherence" metric.
# In "compute" mode, returns the average of the scores given by humans on these types of conversation, instead
def human_conversation(model, tokenizer, filepath, train, length=5):
    # If we are training the metric, go ahead with the chat
    if train:
        # Initialize an empty chat
        chat_history = []
        # Chat for 'length' times
        for step in range(length):
            # Get prompt from user
            user_sentence = input(">> User:")
            # Add the user sentence to the chat history
            chat_history.append(user_sentence)
            # Encode the new user input, add the eos_token and return a tensor
            new_user_input_ids = tokenizer.encode(user_sentence + tokenizer.eos_token, return_tensors='tf')
            # Append the new user input tokens to the chat history
            bot_input_ids = tf.concat([chat_history_ids, new_user_input_ids], axis=-1) if step > 0 else new_user_input_ids
            # Generate a response while limiting the current answer to 128 tokens, using sampling
            max_length = 128 + bot_input_ids.shape[1]
            chat_history_ids = model.generate(bot_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                              do_sample=True, top_p=0.92, top_k=50)
            # Get the last ouput tokens from the bot
            bot_sentence = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            # Append the bot answer to the chat history
            chat_history.append(bot_sentence)
            # Pretty print the bot answer
            print("DialoGPT: {}".format(bot_sentence))
        # Once the chat is over, we ask the user for a score between 0 and 5 (integers only).
        # Initialize variables to get the score
        got_score = False
        score = None
        # While used for input sanity checks on the user input...
        while not got_score:
            # Get the score as a user input
            score = input("How do you rate this conversation (0 to 5)? ")
            # If the score is a valid character, cast it to an integer and proceed
            if score == "0" or score == "1" or score == "2" or score == "3" or score == "4" or score == "5":
                score = int(score)
                got_score = True
            # Otherwise, inform the user that its input is not valid, and ask again
            else:
                print("Invalid score! Must be a single integer between 0 and 5!")
        # Make directories if they do not exists, where to store the csv containing the chats and the user scores
        if not os.path.exists(filepath.rsplit(os.path.sep, 1)[0]):
            os.makedirs(filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
        # If the csv containing chats and user scores exists, first load it, then append a new entry
        if os.path.exists(filepath):
            human_convo_df = pd.read_csv(filepath)
            human_convo_df = human_convo_df.append({"chat":chat_history, "score":score}, ignore_index=True)
        # Otherwise, create it from scratch
        else:
            human_convo_df = pd.DataFrame.from_dict({"chat":[chat_history], "score":score})
        # Save the csv file
        human_convo_df.to_csv(filepath, index=False)
    else:
        # If we only want to compute the metric, read the csv file containing chat-user_scores pairs, and return
        # mean and std of the scores
        human_convo_df = pd.read_csv(filepath)
        return np.average(human_convo_df['score'].to_numpy() / 5), np.std(human_convo_df['score'].to_numpy() / 5)

# Function to run single rounds of question-answer between a chatbot and a pre-set of questions, used to train the style and consistency
# human metrics. In "compute" mode, returns the average of the scores given by humans on these types of queries, instead
def single_answers(model, tokenizer, filepath, train, questions):
    # If we are training the metric, go ahead with the queries
    if train:
        # Initialize an empty query
        questions_history = []
        # Ask each question separately
        for question in questions:
            print("Question: {}".format(question))
            # Encode the question, adding the eos_token at its end
            question_input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='tf')
            # Generate a response while limiting the current answer to 128 tokens, using sampling
            max_length = 128 + question_input_ids.shape[1]
            chat_history_ids = model.generate(question_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id,
                                              do_sample=True, top_p=0.92, top_k=50)
            # Pretty print the bot answer
            bot_sentence = tokenizer.decode(chat_history_ids[:, question_input_ids.shape[-1]:][0], skip_special_tokens=True)
            # Append the question-answer pair to the history
            questions_history.append((question, bot_sentence))
            # Pretty print the bot answer
            print("DialoGPT: {}".format(bot_sentence))
        # Once the queries are over, we ask the user for a score between 0 and 5 (integers only).
        # Initialize variables to get the score
        got_score = False
        score = None
        # While used for input sanity checks on the user input...
        while not got_score:
            # Get the score as a user input
            score = input("How do you rate these answers (0 to 5)? ")
            # If the score is a valid character, cast it to an integer and proceed
            if score == "0" or score == "1" or score == "2" or score == "3" or score == "4" or score == "5":
                score = int(score)
                got_score = True
            # Otherwise, inform the user that its input is not valid, and ask again
            else:
                print("Invalid score! Must be a single integer between 0 and 5!")
        # Make directories if they do not exists, where to store the csv containing the questions and the user scores
        if not os.path.exists(filepath.rsplit(os.path.sep, 1)[0]):
            os.makedirs(filepath.rsplit(os.path.sep, 1)[0], exist_ok=True)
        # Otherwise, create it from scratch
        if os.path.exists(filepath):
            questions_df = pd.read_csv(filepath)
            questions_df = questions_df.append({"questions":questions_history, "score":score}, ignore_index=True)
        else:
            questions_df = pd.DataFrame.from_dict({"questions":[questions_history], "score":score})
        # Save the csv file
        questions_df.to_csv(filepath, index=False)
    else:
        # If we only want to compute the metric, read the csv file containing questions-user_scores pairs, and return
        # mean and std of the scores
        questions_df = pd.read_csv(filepath)
        average_score = np.average(questions_df['score'].to_numpy())
        return np.average(questions_df['score'].to_numpy() / 5), np.std(questions_df['score'].to_numpy() / 5)

# Class defining a wrapper for any of the supported metrics, so that they can be loaded and computed seamlessly
class BBMetric:
    # List of supported metrics
    metrics_list = ["bleu", "semantic similarity", "rouge l",
                    "emotion", "semantic answer similarity", "distinct",
                    "semantic classifier", "perplexity", "human - coherence",
                    "human - consistency", "human - style"]
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
        if name == "bleu":
            self.compute_require_args = set(["predictions", "references"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score']
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
            # Note: This Metric is not autonomous from project folder structure and datatypes! #
            self.compute_require_args = set(["sentences", "character", "character_dict", "base_folder"])
            self.compute_optional_args = set(["n_sentences", "verbose"])
            self.train_require_args = set(["character", "character_dict", "source_dict", "random_state", "base_folder"])
            self.train_optional_args = set(["from_saved_embeddings", "shutdown_at_end", "n_shuffles"])
            self.return_args = ['score', 'std']
        elif name == "perplexity":
            self.compute_require_args = set(["model", "encoded_test_set"])
            self.compute_optional_args = set()
            self.train_require_args = set()
            self.train_optional_args = set()
            self.return_args = ['score']
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
            
    # Pretty print metric
    def __str__(self):
        return str({"instance": self, "name": self.name, "metric": self.metric})
    
    # Function to load a metric, given its name
    def load_metric(name, **kwargs):
        metric = None
        # For each metric, based on the name, we instantiate either a function (algorithmic) or a transformer (neural)
        if name == "bleu":
            metric = BBMetric(name,
                              datasets.load_metric('bleu'))
        elif name == "semantic answer similarity":
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
                              lambda m, s: perplexity(m, s))
        elif name == "semantic similarity":
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

    # Function to compute a metric on sentences/test set
    def compute(self, **kwargs):
        # Check that passed parameters are correct
        if not set(kwargs.keys()).issubset(set(self.compute_require_args).union(set(self.compute_optional_args))):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.compute_require_args)
        if not set(self.compute_require_args).issubset(set(kwargs.keys())):
            raise Exception("Missing arguments! Required arguments are",
                            self.compute_require_args)           
        # Dictionary in which to store results
        result = {}
        if self.name == "bleu":
            # Cast predictions and references as lists, separating tokens, as required by the HuggingFace BLEU metric
            predictions = [x.split() for x in kwargs['predictions']] if type(kwargs['predictions']) is list else [kwargs['predictions'].split()]
            references = [[x.split()] for x in kwargs['references']] if type(kwargs['references']) is list else [[kwargs['references'].split()]]
            # Compute via HuggingFace BLEU metric, and write the resulting score
            self.metric.add_batch(predictions=predictions, references=references)
            result['score'] = self.metric.compute()['bleu']
        elif self.name == "semantic answer similarity":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(kwargs['references']) is list else [kwargs['references']]
            # Pass sentences to model, outputting similarity values
            single_outputs = self.metric.predict(list(zip(predictions, references)))
            # Write mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "rouge l":
            # Cast predictions and references as lists
            predictions = kwargs['predictions'] if type(kwargs['predictions']) is list else [kwargs['predictions']]
            references = kwargs['references'] if type(kwargs['references']) is list else [kwargs['references']]
            single_outputs = []
            # Compute separately for each prediction-reference pair the Rouge-L metric. Get the average f-measure
            for i in range(len(predictions)): 
                self.metric.add(prediction=predictions[i],
                                reference=references[i])
                single_outputs.append(self.metric.compute()['rougeL'].mid.fmeasure)
            # Write mean and std of these scores  
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "emotion":
            # Cast sentences as a list
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            # Pass sentences through the metric, outputting scores for each emotion label
            output = self.metric(sentences)
            result['score'] = {}
            result['std'] = {}
            # Initialize lists of scores for each emotion
            for emotion_dict in output[0]:
                result['score'][emotion_dict['label']] = []
            emotions_n = len(result['score'])
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
                result['std'][emotion] = np.std(np.array(result['score'][emotion]))
                result['score'][emotion] = np.mean(np.array(result['score'][emotion]))
            # Return a dictionary with lists for labels, avg scores and stds, in corresponding order
            result['label'] = list(result['score'].keys())
            result['score'] = list(result['score'].values())
            result['std'] = list(result['std'].values())
        elif self.name == "semantic similarity":
            # Cast sentences as lists
            sentences_a = kwargs['sentences_a'] if type(kwargs['sentences_a']) is list else [kwargs['sentences_a']]
            sentences_b = kwargs['sentences_b'] if type(kwargs['sentences_b']) is list else [kwargs['sentences_b']]
            # Pass sentences through the model separately, then apply cosine similarity between the two encoded vectors, and
            # get diagonal values only (comparison between sentences_a_i and sentences_b_i for all i)
            single_outputs = np.diagonal(cosine_similarity(self.metric.encode(sentences_a),
                                                           self.metric.encode(sentences_b)))
            # Compute mean and std of these scores
            result['score'] = np.mean(np.array(single_outputs))
            result['std'] = np.std(np.array(single_outputs))
        elif self.name == "distinct":
            # Cast sentences as a list
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            # Compute distinct, obtaining mean and std for the metric
            result['score'], result['std'] = self.metric(sentences,
                                                         kwargs['ngram_size'] if 'ngram_size' in kwargs else 3)
        # Compute perplexity or human metrics by simply passing params
        elif self.name == "perplexity":
            result['score'] = self.metric(kwargs['model'], kwargs['encoded_test_set'])
        elif self.name == "human - coherence":
            result['score'], result['std'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "human - consistency":
            result['score'], result['std'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "human - style":
            result['score'], result['std'] = self.metric(None, None, kwargs['filepath'], False, None)
        elif self.name == "semantic classifier":
            # Cast sentences as a list
            sentences = kwargs['sentences'] if type(kwargs['sentences']) is list else [kwargs['sentences']]
            # Assert there are enough sentences for the semantic classifier to perform a single evaluation
            if len(sentences) < 3:
                raise Exception("Needs at least three sentences to run the classifier!")
            # Perform sanity check on 'n_sentences' parameter
            n_sentences = kwargs['n_sentences'] if 'n_sentences' in kwargs else 'all'
            if n_sentences != 'all' and type(n_sentences) != int:
                raise Exception("Invalid type for n_sentences!")
            if type(n_sentences) == int and n_sentences < 0:
                raise Exception("Invalid number of sentences!")
            if type(n_sentences) == int and n_sentences > len(sentences):
                n_sentences = 'all'
            # Compute the semantic classifier metric, returning scores for each sentences triple
            outputs = self.metric.compute(sentences, kwargs['character'], kwargs['character_dict'], kwargs['base_folder'],
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
        if not set(kwargs.keys()).issubset(set(self.train_require_args).union(set(self.train_optional_args))):
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