Results in this folder are obtained with distilbert classifier trained on 3 steps of 2 epochs each.
Given the sentence at index "i" from the dataset, the input of the distilbert classifier is the concatenation of lines from index "i-n" to "i+n", 
where "n" is the parameter studied.
All models are saved in the parent folder "Metrics/Distilbert-Embedded Chatbot Classifier" with names "distilbert_embedder_n_*".