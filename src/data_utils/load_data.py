from os.path import join
from os import listdir
from tqdm import tqdm
import pandas as pd
import re
from datasets import DatasetDict
from datasets import load_dataset

from .data_dicts import source_dict, character_dict, random_state


# Open the dataset documents and store their data into a DataFrame
def load_char_dataset(character, base_folder):
    ### Loading functions from other files
    # Load the dataset from How I Met Your Mother or
    #                       The Big Bang Theory   or
    #                       Friends
    def _load_himym_friends_tbbt_dataset(sources_folder):
        dataframe_rows = []
        # Get number of documents and their names
        documents_n = len(listdir(sources_folder))
        documents_names = listdir(sources_folder)
        # Loop over documents
        for i in tqdm(range(documents_n)):
            # Extract filename which correspond to the link of the episode
            filename = documents_names[i]
            # the last 5 chars takes the form `sxe` with s the number of the current serie and
            # and e as the number of the episode
            sources_label = filename[:-4]
            # Open document
            with open(join(sources_folder, filename), encoding="utf8") as file:
                # Loop over lines (= words)
                for line in file.readlines():
                    dataframe_row = {
                        "source": sources_label,
                        "line": line,
                    }
                    dataframe_rows.append(dataframe_row)
        # Build the dataframe from the words
        df = pd.DataFrame(dataframe_rows)
        return df

    # Load the dataset from Futurama
    def _load_futurama_dataset(sources_folder):
        futurama_txt = ''
        # Loop over documents
        for filename in tqdm(listdir(sources_folder)):
            futurama_txt += open(join(sources_folder, filename),
                                 encoding='utf-8').read()
        # Split lines
        start_idx = 0
        end_idx = 0
        lines = []
        while start_idx < len(futurama_txt):
            # eventually bold tag are present, discard them
            start_idx = futurama_txt.find('<b>', end_idx)
            if start_idx == -1:  # if no '<b>' is found, just save the rest
                lines.append(futurama_txt[end_idx:].replace('</b>', ''))
                break
            elif start_idx != end_idx:  # '<b>' is found
                lines.append(futurama_txt[end_idx + 4:start_idx])
            end_idx = futurama_txt.find('</b>', start_idx)
            if end_idx == -1:  # if no '</b>' is found, just save the rest
                lines.append(futurama_txt[start_idx:].replace('<b>', ''))
                break
            lines.append(futurama_txt[start_idx + 3:end_idx])
        df = pd.DataFrame(lines, columns=['line'])
        return df

    # Load the dataset from Harry Potter
    def _load_hp_dataset(sources_folder):
        sep = ';'
        df = None
        df_files = []
        # for each movie append the dataset which refers to it
        for filename in listdir(sources_folder):
            df_files.append(
                pd.read_csv(join(sources_folder, filename),
                            sep=sep).rename(columns=lambda x: x.lower()))
        df = pd.concat(df_files)
        df = df.rename(columns={'character': 'character', 'sentence': 'line'})
        return df

    # Load the dataset from Star Wars
    def _load_sw_dataset(source_folder):
        dataframe_rows = []
        # Get number of documents and their names
        documents_n = len(listdir(source_folder))
        documents_names = listdir(source_folder)
        # Loop over documents
        for i in tqdm(range(documents_n)):
            filename = documents_names[i]
            film_name = filename[:-4]
            # Open document
            with open(join(source_folder, filename), encoding='utf-8') as file:
                film_rows = []
                sentence = ""
                empty_line_allow = False
                between_numbers = False
                found_character = False
                for line in file.readlines():
                    if re.search(
                            r"^[0-9]+.", line
                    ) != None:  # Line is number followed by dot (page number)
                        pass
                    elif re.search(
                            r"^[A-Z]{2,}", line
                    ) != None:  # Line begins with an-all caps (a character)
                        sentence += line
                        found_character = True
                        empty_line_allow = True
                    elif line.isspace():
                        if empty_line_allow:
                            pass
                        else:
                            if found_character:
                                film_row = {
                                    "film": film_name,
                                    "line": sentence,
                                }
                                film_rows.append(film_row)
                                sentence = ""
                                found_character = False
                    elif found_character:
                        sentence += line
                        empty_line_allow = False
                dataframe_rows.extend(film_rows)
        # Build the dataframe from the words
        df = pd.DataFrame(dataframe_rows)
        return df

    ### Function starts here
    # if character selected is 'Default' so we don't need any dataset
    if character == 'Default':
        # no dataset is loaded
        return None
    # otherwise let's take from the source dictionary the folder which contains the datasets
    # sources_subfolder is a parameter which contains the path where all data are stored, it can
    #   be different from null if data are stored in a different subfolder
    source = character_dict[character]['source']
    sources_subfolder = source_dict[source]['dataset_folder']
    if sources_subfolder:
        sources_folder = join(base_folder, "Data", "Sources", source,
                              sources_subfolder)
    else:
        sources_folder = join(base_folder, "Data", "Sources", source)
    # each tv shows loads data by a call to its respective function
    if source == 'HIMYM' or source == 'Friends' or source == 'TBBT':
        df = _load_himym_friends_tbbt_dataset(sources_folder)
    elif source == 'Futurama':
        df = _load_futurama_dataset(sources_folder)
    elif source == 'HP':
        df = _load_hp_dataset(sources_folder)
    elif source == 'SW':
        df = _load_sw_dataset(sources_folder)
    return df


# Function used to load the dataframe of the character selected
def load_df(character, base_folder):
    """
    docstring
    """
    # Takes the folder of the character
    dataset_path = join(base_folder, "Data", "Characters", character,
                        character + '.csv')

    # Load HuggingFace dataset
    character_hg = load_dataset('csv',
                                data_files=dataset_path,
                                cache_dir=join("cache"))

    # Perform 85% train / 10% test / 5% validation with a fixed seed
    train_test_hg = character_hg['train'].train_test_split(test_size=0.15,
                                                           seed=random_state)
    test_val = train_test_hg['test'].train_test_split(test_size=0.33,
                                                      seed=random_state)

    # Store splits into a HuggingFace dataset
    character_hg = DatasetDict({
        'train': train_test_hg['train'],
        'test': test_val['train'],
        'val': test_val['test']
    })
    return character_hg
