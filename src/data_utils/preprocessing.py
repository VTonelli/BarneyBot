import pandas as pd
import json

from os.path import join, exists
from tqdm import tqdm
from .data_dicts import model_name, character_dict
from transformers import AutoTokenizer

# Parameter for generation method Beam Search
n_beams = 3
# Parameters for generation method Sampling
top_k = 50
top_p = 0.92


def get_dataframe_for_metrics(data_test, predictions_greedy,
                              predictions_nbeams, predictions_sampling,
                              tokenizer):
    """
    docstring
    """

    i = 0
    # Dataframe initialized containing two columns: 1 for explicit context and 1 for its tokenized version
    df = {'ctx': [], 'ctx_tk': []}
    # If real character lines from the script are feature of test set
    has_labels = 'response' in data_test.features
    if has_labels:
        # Add the explicit and tokenized labels columns to the Dataframe
        df['lbl'] = []
        df['lbl_tk'] = []
    # If predictions by Greedy generation method are requested
    if predictions_greedy:
        # Add the explicit and tokenized predictions columns with this method to the Dataframe
        df['prd_greedy'] = []
        df['prd_greedy_tk'] = []
    # If predictions by Beam Search generation method are requested
    if predictions_nbeams:
        # Add the explicit and tokenized predictions columns with this method to the Dataframe
        df['prd_nbeams'] = []
        df['prd_nbeams_tk'] = []
    # If predictions by Sampling generation method are requested
    if predictions_sampling:
        # Add the explicit and tokenized predictions columns with this method to the Dataframe
        df['prd_sampling'] = []
        df['prd_sampling_tk'] = []
    # Iterate over each sample in test set
    for sample in tqdm(data_test):
        # Encode the context and label sentences, add the eos_token and return a tensor
        ctx_tk = tokenizer.encode(sample['context/0'] + tokenizer.eos_token,
                                  return_tensors='tf').numpy().tolist()
        ctx = sample['context/0']
        df['ctx_tk'].append(ctx_tk)
        df['ctx'].append(ctx)
        # Labels
        if has_labels:
            lbl_tk = tokenizer.encode(sample['response'] + tokenizer.eos_token,
                                      return_tensors='tf').numpy().tolist()
            lbl = sample['response']
            df['lbl'].append(lbl)
            df['lbl_tk'].append(lbl_tk)
        # Greedy
        if predictions_greedy:
            prd_greedy_tk = predictions_greedy[i]
            prd_greedy = tokenizer.decode(prd_greedy_tk,
                                          skip_special_tokens=True)
            df['prd_greedy'].append(prd_greedy)
            df['prd_greedy_tk'].append(prd_greedy_tk)
        # Beam Search
        if predictions_nbeams:
            prd_nbeams_tk = predictions_nbeams[i]
            prd_nbeams = tokenizer.decode(prd_nbeams_tk,
                                          skip_special_tokens=True)
            df['prd_nbeams'].append(prd_nbeams)
            df['prd_nbeams_tk'].append(prd_nbeams_tk)
        # Sampling
        if predictions_sampling:
            prd_sampling_tk = predictions_sampling[i]
            prd_sampling = tokenizer.decode(prd_sampling_tk,
                                            skip_special_tokens=True)
            df['prd_sampling'].append(prd_sampling)
            df['prd_sampling_tk'].append(prd_sampling_tk)
        i += 1
    return pd.DataFrame(data=df)


#


# Function to construct a conversation from the rows of a dataset
def construct_conv(row, tokenizer):
    """
    docstring
    """
    # Max conversation length
    MAX_LENGTH = 512
    # Reverse the rows, since they are originally in order response->context/0->context/1...
    row = list(reversed(list(row.values())))
    # Pass row into the tokenizer, getting a dictionary with input_ids and attention_mask
    model_inputs = tokenizer(row)
    # Get pad token encoding
    tokenizer_pad_token_id = tokenizer.encode('#')[0]
    # Append to each row element the eos token, to separate the sentences
    for i in range(len(model_inputs['input_ids'])):
        model_inputs['input_ids'][i].append(tokenizer.eos_token_id)
        model_inputs['attention_mask'][i].append(1)
    # Transform the lists into a single concatenated conversation
    model_inputs['input_ids'] = [
        item for sublist in model_inputs['input_ids'] for item in sublist
    ]
    model_inputs['attention_mask'] = [
        item for sublist in model_inputs['attention_mask'] for item in sublist
    ]
    # If there is extra space, append padding tokens with attention mask 0
    if MAX_LENGTH > len(model_inputs['input_ids']):
        model_inputs['input_ids'] += [tokenizer_pad_token_id] * (
            MAX_LENGTH - len(model_inputs['input_ids']))
        model_inputs['attention_mask'] += [0] * (
            MAX_LENGTH - len(model_inputs['attention_mask']))
    # If on the other hand the conversation is too long, truncate it, always setting eos as the last token
    elif MAX_LENGTH < len(model_inputs['input_ids']):
        model_inputs['input_ids'] = model_inputs['input_ids'][:MAX_LENGTH - 1]
        model_inputs['input_ids'][-1] = tokenizer.eos_token_id
        model_inputs['attention_mask'] = model_inputs[
            'attention_mask'][:MAX_LENGTH - 1]
        model_inputs['attention_mask'][-1] = 1
    # Since dialogpt is an autoregressive model, labels should be equal to inputs during training
    model_inputs["labels"] = model_inputs["input_ids"]
    # Return the dictionary
    return model_inputs


#


# Function defining the formatting of the dataset rows, so that they can be fed to dialogpt
# (DialoGPT requires a conversation to be fed as a single string of concatenated exchanges)
def preprocess_function(examples):
    """
    docstring
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=join("cache"))
    tokenizer.pad_token = '#'

    # Run the function to construct conversations defined above
    model_inputs = construct_conv(examples, tokenizer)
    return model_inputs


#


# Function that extract predictions from a stored file to speed up the process
def get_predictions_cached(sample_questions,
                           model,
                           filename,
                           generation_method,
                           character,
                           tokenizer,
                           base_folder,
                           override_predictions=False):

    prediction_path = join(base_folder, 'Data', 'Characters', character,
                           filename)
    # If the predictions have already been created
    if exists(prediction_path) and not override_predictions:
        # It loads them
        print("Loading predictions from stored file")
        with open(prediction_path, 'r', encoding='utf-8') as file:
            json_string = file.read()
        predictions = json.loads(json_string)
        print("Loaded predictions from stored file")

    else:
        # Otherwise they are created
        print("Creating predictions")
        predictions = list()
        for x in tqdm(sample_questions):
            tokenized_question = tokenizer.encode(x + tokenizer.eos_token,
                                                  return_tensors='tf')
            # Max length of each tokenized sequence must be the following
            max_length = 128 + tokenized_question.shape[1]
            if generation_method == "Greedy":  # Greedy generation method
                generated_answer = model.generate(
                    tokenized_question,
                    pad_token_id=tokenizer.eos_token_id,
                    max_length=max_length)[0].numpy().tolist()
            elif generation_method == "Beam Search":  # Beam Search generation method
                generated_answer = model.generate(
                    tokenized_question,
                    pad_token_id=tokenizer.eos_token_id,
                    max_length=max_length,
                    n_beams=n_beams)[0].numpy().tolist()
            elif generation_method == "Sampling":  # Sampling generation method
                b = True
                c = 0
                while b:
                    generated_answer = model.generate(
                        tokenized_question,
                        pad_token_id=tokenizer.eos_token_id,
                        max_length=max_length,
                        do_sample=True,
                        top_k=top_k,
                        top_p=top_p)[0].numpy().tolist()
                    c += 1
                    if len(generated_answer[len(tokenized_question[0]):]) > 1:
                        b = False
                    if c > 100:
                        generated_answer[len(tokenized_question[0]
                                             ):] = tokenizer.encode('hi') + [
                                                 tokenizer.eos_token_id
                                             ]
                        break
            # Append predictions
            predictions.append(generated_answer[len(tokenized_question[0]):])

        # Save predictions as a JSON file
        output_string = json.dumps(predictions)
        with open(prediction_path, 'w', encoding='utf-8') as file:
            file.write(output_string)

        assert all([len(p) > 1 for p in predictions])

    return predictions


#


def process_dataset(df, character):

    def _process_himym_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        df = df[~df['line'].isnull()]
        df[['character', 'line']] = df['line'].str.split(":", 1, expand=True)
        # Removes empty lines
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_tbbt_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        df = df[~df['line'].str.startswith("Scene: ")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df[['character', 'line']] = df['line'].str.split(":", 1, expand=True)
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_futurama_dataset(df):
        # Remove white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round and square brackets
        df['line'] = df['line'].str.replace(r"\[.*\]", "")
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes everything is inside the tags
        df['line'] = df['line'].str.replace(r"\<.*\>", "")
        df['line'] = df['line'].str.replace(r"\s+", " ")
        df['line'] = df['line'].str.replace("\n", "")
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("(")]
        df = df[~df['line'].str.startswith("[")]
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        df_rows = []
        for row in tqdm(range(len(df) - 1)):
            if df['line'][row].isupper():
                df_row = {
                    'line': df['line'][row + 1].strip()[:512],
                    'character': df['line'][row].strip().capitalize()
                }
                df_rows.append(df_row)
        df = pd.DataFrame(df_rows)
        # Discard titles
        df = df[df['character'].str.contains('Futurama') == False]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_friends_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        df = df[~df['line'].isnull()]
        df[['character', 'line']] = df['line'].str.split(":", 1, expand=True)
        # Removes empty lines
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['line'] = df['line'][df['line'].str.len() >= 2]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df[~(df['character'] == 'Written by')]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_sw_dataset(df):
        # Removes lines which starts with brackets
        df = df[~df['line'].str.startswith("[")]
        df = df[~df['line'].str.startswith("(")]
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        df[['character', 'line']] = df['line'].str.split("\n", 1, expand=True)
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df[df['character'].str.split().apply(lambda l: len(l)) <= 6]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def _process_hp_dataset(df):
        # Removes white space
        df['line'] = df['line'].str.strip()
        # Removes everything is inside the round brackets
        df['line'] = df['line'].str.replace(r"\(.*\)", "")
        # Removes bracket char, newline, tabular char and special chars replacing them with a space
        df['line'] = df['line'].str.replace(r"[\/(){}\[\]\|@_#]|\\t|\\n", " ")
        # Removes every char which is not present in the following "white list"
        df['line'] = df['line'].str.replace(r"[^.\',;:?!0-9a-zA-Z \-]", "")
        # Remove empty lines
        df = df[~df['line'].isnull()]
        df = df.dropna()
        # Removes white space
        df['line'] = df['line'].str.strip()
        df['character'] = [line.lower() for line in df['character']]
        # Removes empty lines
        df = df[~df['line'].isnull()]
        df = df.replace(r'^s*$', float('NaN'), regex=True)
        # Removes empty lines
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    # Function starts here
    if character == 'Default':
        return None
    source = character_dict[character]['source']
    if source == 'HIMYM':
        df = _process_himym_dataset(df)
    elif source == 'Friends':
        df = _process_friends_dataset(df)
    elif source == 'Futurama':
        df = _process_futurama_dataset(df)
    elif source == 'TBBT':
        df = _process_tbbt_dataset(df)
    elif source == 'HP':
        df = _process_hp_dataset(df)
    elif source == 'SW':
        df = _process_sw_dataset(df)
    return df
