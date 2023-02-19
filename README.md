# Project of Natural Language Processing: BarneyBot

## Abstract
We developed several chabots using the pretrained model of DialoGPT from transformer library of 🤗 Hugginface by performing fine-tuning of its small version on some corpus of data coming from tv show and movies, for different characters.

We choose to extend the work made by [Nguyen et al.] (https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761115.pdf) who explored the task by implementing a chatbot by a seq-to-seq model. This work is by all means a revision and extension of theirs.

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
Please install all dependencies within "requirements.txt" through pip. There is also a GPU version for these
same requirements, but it may is CUDA dependant.

## Repository structure
The list of relevant folders for this repository is:
* `Data` folder contains all the data we used to fine-tune our models and where we saved the models,
* `Code` folder contains the notebooks and also the custom libraries useful to compute metrics and plotting,
* `Metrics` folder which contains metric results in json format and plots

## Metric Evaluation
As an extension of the original project, we implemented and tested a variety of metrics to evaluate the performances of these chatbots.

## Additional Data
We also deployed a drive folder which already contains the trained models, you can check it following this [drive link](https://liveunibo-my.sharepoint.com/:f:/g/personal/valerio_tonelli2_studio_unibo_it/EqAjIM0kvAJFjrNXPPnSLe4BVC4cAMsfrr-7s_SjUiWDrg)