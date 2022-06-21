[logoHF]: https://huggingface.co/front/assets/huggingface_logo-noborder.svg "Logo HF"
[logoHF1]: [https://huggingface.com/%F0%9F%A4%97] "Logo HF1"

# Project of Natural Language Processing: BarneyBot

## Abstract
We developed a chabot using pretrained model of DialoGPT from transformer library of 🤗 Hugginface performing a fine tuning of its small version on some corpus of data coming from tv show or movie saga.

We choise to extend the work made by [Nguyen et al.](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761115.pdf) who explored the task implementing a chatbot by a seq-to-seq model. As previously said we decided to approch the problem by implementing a more sophisticated model architecture i.e. DialoGPT. 

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
