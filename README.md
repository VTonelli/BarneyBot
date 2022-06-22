# Project of Natural Language Processing: BarneyBot

## Abstract
We developed a chabot using pretrained model of DialoGPT from transformer library of 🤗 Hugginface performing a fine tuning of its small version on some corpus of data coming from tv show or movie saga.

We choise to extend the work made by [Nguyen et al.](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761115.pdf) who explored the task implementing a chatbot by a seq-to-seq model. As previously said we decided to approch the problem by implementing a more sophisticated model architecture i.e. DialoGPT. 

### Datasets:
| Character      | TV show/movie         |
|----------------|-----------------------|
| Barney Stinson | How I Met Your Mother |
| Sheldon Cooper | The Big Bang Theory   |
| Joey           | Friends               |
| Phoeby         | Friends               |
| Harry Potter   | Harry Potter          |
| Fry            | Futurama              |
| Bender         | Futurama              |
| Darth Vader    | Star Wars             |

## Initial setup
To be able to run the project an initial setup is required. More details about it can be found in 🤗 Transformers [user guide](https://github.com/huggingface/transformers).

You should install 🤗 Transformers in a virtual environment. If you're unfamiliar with Python virtual environments, check out the user guide.

* First thing to do is to create a virtual environment with the specified version of Python.
* Then as required by the user guide of Transformers, you will need to install at least one of Flax, PyTorch or TensorFlow.
    - When one of those backends has been installed, 🤗 Transformers can be installed using pip as follows
    ```
    pip install transformers
    ```
    - or by using the conda command, by selecting the channel `huggingface`
    ```
    conda install -c huggingface transformers
    ```
* Another requirement to execute correctly the project is the need to setup the package 🤗 Datasets. 🤗 Datasets is a library for easily accessing and sharing datasets, and evaluation metrics for Natural Language Processing (NLP), computer vision, and audio tasks. More information about installation are in this [user guide](https://huggingface.co/docs/datasets/installation).
    - The most straightforward way to install 🤗 Datasets is with pip:
    ```
    pip install datasets
    ```
    - or by using the conda command
    ```
    conda install -c huggingface -c conda-forge datasets
    ```
* Other relevant dependencies are the following:
    - [SentenceTransformers](https://www.sbert.net/) which is a Python framework for state-of-the-art sentence, text and image embeddings. You can install it using pip:
    ```
    pip install -U sentence-transformers
    ```
    - [Rouge score]() that is a native python implementation of ROUGE score, designed to replicate results from the original perl package. It can be installed in the following way:
    ```
    pip install rouge/requirements.txt
    pip install rouge-score
    ```

## Repository structure
The list of relevant folders for this repository is.
* `Data` folder contains all the data we used to fine tune our models,
* `Lib` folder contains library useful to compute metrics and plotting,
* `Metrics` folder which contains metric results in json format and plots

The list of notebooks is instead:
* `Preprocessing` where the preprocessing data is performed
* `Bot` in this notebook is performed the real fine tune of the models
* `Build Common Dataset` which allows us to construct a common dataset composed by a set of pre-selected sentences from each corpus
* `Evaluation` where all the metrics selected are computed 
* `Visualization` which has the goal to print all the plots of the selected metrics 
* `Human Metrics` a notebook aimed to allow external users to take part of the evaluation by humans metrics 

## Running tutorial
Assuming you want run the project for the character _Barney Stinson_
1. run `Preprocessing` notebook, to preprocess and clean up the dataset, with the following setup:
   ```
   character = 'Barney'
   ```
2. run `Bot` notebook, to fine tune the chatbot for making it speak like Barney, with the following setup:
   ```
   character = 'Barney'
   ```
3. run `Evaluation`, to evaluate the chatbot fine tuned, with the following setup:
   ```
   character = 'Barney'
   ```
   and ranning all the cells in the notebook under the following titles:
   - Metrics For Character 1
   - Metrics Between Different Sampling Methods
   - Metrics Between Character vs Non-Finetuned
4. Finally to see the results of the evaluation you can run `Visualization`

Assuming you would like to perform some comparison between two character, e.g. _Barney_ vs _Sheldon_ you can (alternatively to 3. and 4. steps):
1. run `Build Common Dataset` for building a common dataset to use for chatbots comparison
2. run all the `Evaluation`, with instead the following setup:
   ```
   character = 'Barney'
   character_2 = 'Sheldon'
   ```