import hashlib
import json
from enum import Enum
import os

from .BBData import character_dict

class MetricDependency(int, Enum):
    DATASET = 0  # Metric depends on datasets only and/or base DialoGPT model
    COHERENT = 1  # Metric depends on chatbot trained on its dataset
    ADVERSARIAL = 2  # Metric depends on chatbot trained on a dataset but with another dataset of reference
    COMPARATIVE = 3  # Metric depends on comparison between chatbots


class MetricArity(int, Enum):
    SINGLE = 1  # Metric depends on a single actor
    PAIRWISE = 2  # Metric depends on two actors


class MetricDeterminism(int, Enum):
    DETERMINISTIC = 0  # There is a closed-form equation for this metric, which is fully computed
    PROBABILISTIC = 1  # The metric is obtained through explainable approx., e.g. SGD, partial computation on a subset...
    NEURAL = 2  # The metric is obtained via a neural network
    HUMAN = 4  # The metric is obtained via human surveying


class MetricActor(int, Enum):
    DATASET_CHARCONTEXT = 0  # Context sentences [any character but not 'Default', including "Common"]
    DATASET_CHAR = 1  # Labels or the entire dataset [any character but not 'Default', including "Common"]
    DIALOGPT_GREEDY = 10  # [any character including 'Base']
    DIALOGPT_NBEAMS = 11  # [any character including 'Base']
    DIALOGPT_SAMPLE = 12  # [any character including 'Base']

def is_character(char):
    if char in character_dict.keys() and char != 'Default':
        return True
    elif char == 'Base' or char == 'Common':
        return False
    else:
        raise Exception("Unknown character name " + char + "!")

def save_as_json(filepath, filename, data):
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename + ".json"),
              'w',
              encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4))

def load_from_json(filepath, filename):
    if not os.path.exists(os.path.join(filepath, filename + '.json')):
        return dict()
    with open(os.path.join(filepath, filename + '.json'),
              'r',
              encoding='utf-8') as f:
        return json.load(f)

def dict_hash(dictionary):
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def get_metric_arity(metric_name):
    if metric_name == 'bleu' or metric_name == 'rouge l' or metric_name == 'semantic similarity' or \
       metric_name == 'semantic answer similarity' or metric_name == 'chatbot classifier' or metric_name == 'perplexity':
        return MetricArity.PAIRWISE
    elif metric_name == 'distinct' or metric_name == 'emotion classifier' or metric_name == 'lines count':
        return MetricArity.SINGLE
    elif metric_name == 'dummy metric':
        return MetricArity.PAIRWISE
    else:
        raise Exception("Unknown arity for metric " + metric_name)

def get_metric_determinism(metric_name, metric_version):
    if metric_name == 'lines count' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'bleu' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'rouge l' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'semantic similarity' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'semantic answer similarity' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'perplexity' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'chatbot classifier' and metric_version == 1:
        return MetricDeterminism.PROBABILISTIC
    elif metric_name == 'distinct' and metric_version == 1:
        return MetricDeterminism.DETERMINISTIC
    elif metric_name == 'emotion classifier' and metric_version == 1:
        return MetricDeterminism.NEURAL
    elif metric_name == 'dummy metric':
        return MetricDeterminism.DETERMINISTIC
    else:
        raise Exception("Unknown determinism for metric " + metric_name)

def get_metric_dependency(metric_name, metric_actors):
    actors_order = [
        'training_set', 'predictor', 'reference', 'document', 'document0',
        'document1'
    ]
    actor_types = [
        metric_actors[key][0] for key in actors_order if key in metric_actors
    ]
    actor_chars = [
        metric_actors[key][1] for key in actors_order if key in metric_actors
    ]
    if metric_name == 'lines count' or metric_name == 'distinct' or metric_name == 'emotion classifier':
        if all(at.value < 10 for at in actor_types):
            return MetricDependency.DATASET
        elif actor_chars[0] == 'Base':
            return MetricDependency.DATASET
        else:
            return MetricDependency.COHERENT
    elif metric_name == 'bleu' or metric_name == 'rouge l' or \
         metric_name == 'semantic answer similarity' or metric_name == 'chatbot classifier' or metric_name == 'perplexity':
        if all(at.value < 10 for at in actor_types):
            return MetricDependency.DATASET
        elif actor_types[0].value < 10 and actor_chars[1] == 'Base':
            return MetricDependency.DATASET
        elif actor_chars[0] == actor_chars[1]:
            return MetricDependency.COHERENT
        elif all(at.value >= 10 for at in actor_types):
            return MetricDependency.COMPARATIVE
        else:
            return MetricDependency.ADVERSARIAL
    elif metric_name == 'dummy metric':
        return MetricDependency.DATASET
    else:
        raise Exception("Unknown dependency for metric " + metric_name)

def save_metric_by_name(path, filename, metric_dict):
    if os.path.exists(os.path.join(path, filename)):
        metrics = load_from_json(path, filename)
    else:
        metrics = dict()
    metrics.update(metric_dict)
    save_as_json(path, filename, metrics)

def load_metric_by_name(path, filename):
    metrics = load_from_json(path, filename)
    for entry in metrics.values():
        for actor in entry['metric_actors'].values():
            actor[0] = MetricActor(actor[0])
        entry['metric_dependency'] = MetricDependency(
            entry['metric_dependency'])
        entry['metric_determinism'] = MetricDependency(
            entry['metric_determinism'])
        entry['metric_arity'] = MetricArity(entry['metric_arity'])
    return metrics