# Joint entity recognition and relation extraction as a multi-head selection problem

MultiHead Joint Entity Relation Extraction

Input  : Sentence, Entity 1, Entity1_type, Entity1_start_position, Entity1_end_position,Entity 2,Entity2_type, Entity2_start_position, Entity2_end_position,relation

Output : sentence, Entity1, Entity2, Predicted Relation, True Relation

## Approach 

The input of our model is the words of a sentence which are then represented as word vectors(i.e., embeddings).
The BiLSTM layer extracts a more complex representation of each word.
Then the CRF and the sigmoid layers are able to produce the outputs for the two tasks.


![alt text](https://github.com/Sanjithae/Joint_NER_RE/blob/master/Figure1.PNG)

The characters of each word are represented by character vectors that are learned during training. The character embeddings are fed into a BiLSTM and the two final states are concatenated. We take the representation for each word which is the character level representation of the word and then concatenate it to form the word level representation(word2vec) to obtain the complete word embedding vector.

## Training format the model needs to work

![alt text](https://github.com/Sanjithae/Joint_NER_RE/blob/master/Figure2.PNG)

## Configuration

The model has several parameters such as:

EC (entity classification) or BIO (BIO encoding scheme)
Character embeddings
Ner loss (softmax or CRF)
that could be specified in the configuration files.

## Run the model

Download the zip file from the google drive link in the Joint_NER_RE/data/CoNLL04/ and place the zip file at that location.

python main.py

When running for the first time run the following commands and then run python main.py

import nltk

nltk.download('averaged_perceptron_tagger')

Code currently runs on CoNLL04 config file by default.

For other detailed explanation of how to modify the code to run for your dataset (https://github.com/Sanjithae/Joint_NER_RE/blob/master/How_to_run.txt)



## Benchmark Datasets
1) CoNLL04
2) SemEval 2010 Task 8
3) NYT
4) DDI2013

## Evaluation Metrics
1) Precision
2) Recall
3) F1 Score

## Results 

 Test Score is 70.766 in 150 epoch on CoNLL04.
 
 Test Score is 68.95 in 150 epoch on SemEval 2010 Task 8.
 
 
 ## Original Code 
 
 https://github.com/bekou/multihead_joint_entity_relation_extraction
 

## Jupyter Notebook
You can see the execution of the code in this [jupyter notebook](https://github.com/Sanjithae/Joint_NER_RE/blob/master/Joint_NER_RE_Demo.ipynb)


## Youtube Video

Click [here](https://youtu.be/8sQ357ymC_U) to play the video.

## Citation
 
Giannis Bekoulis, Johannes Deleu, Thomas Demeester, Chris Develder. Joint entity recognition and relation extraction as a multi-head selection problem. Expert Systems with Applications, Volume 114, Pages 34-45, 2018

Link to the paper : https://arxiv.org/abs/1804.07847

