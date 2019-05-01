# Python 3 code for AtNRE (Adversarial Training for Relation Extraction)

## Yi Wu, David Bamman, Stuart Russell, ”Adversarial Training for Relation Extraction", Conference on Empirical Methods in Natural Language Processing (EMNLP) 2017, Copenhagen, Denmark

This is a refactored implementation of https://github.com/jxwuyi/AtNRE

A novel method for relation extraction by adapting the well-known Adversarial Training framework for CNN/RNN

Paper Link:  https://people.eecs.berkeley.edu/~russell/papers/emnlp17-relation.pdf

## Input/Output format for training and prediction

Input: .txt files with entity2id, relation2id, test, train, vec files

Output: .pkl file with best prediction of relation between all 2 entities in the test set

## Common Input/Output format for training and prediction /ditk

Input: Sentence, Entity Pairs, Entity Position, Relation

Output: Relation with Highest Probability 

## Approach – ADVERSARIAL training / CNN + RNN  

Input dataset [MIML framework], pretrained embeddings in MIML framework

Construct P-CNN and RNN [Dropout is applied] 

Piecewise – CNN [Piecewise Pooling] , RNN [Tradeoffs!]

Perform Adversarial Training [Introduce Noise]

Test & Predict

## Architecture 

![alt text](https://github.com/aru-jo/ditk/blob/develop/extraction/relation/AtNRE/readme-images/architecture.png)

## Running the code

Please refer to gain/demo/AtNRE.ipynb to help you start the process 
    
	Create a AtNRE object 
  
  Make sure you have the train and vec directory
    directory = AtNRE_dir + '/origin_data/train.txt'
    vec_dir = AtNRE_dir + '/origin_data/vec.txt'
	
	Make sure data is present in origin_data folder or the path you are specifiying 
	  Invoke obj.read_dataset(directory) to read dataset
    Invoke obj.load_embedding(vec_dir) to load embedding
	  Invoke obj.train() to train 
    Invoke obj.predict() to predict 
	  Invoke obj.evaluate() to evaluate

## Data 

We use the same dataset(NYT10) as in [Lin et al.,2016]. And we provide it in origin_data/ directory.

Pre-Trained Word Vectors are learned from New York Times Annotated Corpus (LDC Data LDC2008T19), which should be obtained from [data]. And we provide it also in the drive link
:https://drive.google.com/drive/folders/1lDaXo6-qFBvjd-oQQ-PUNmH0XBqAa5Y-?usp=sharing.

Entity embeddings are randomly initialized. The number of entities in the entity embedding should be the same with the number of entities in train.txt.

To run our code, the dataset should be put in the folder origin_data/ using the following format, containing five files.

train.txt: training file, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).
test.txt: test file, same format as train.txt.
relation2id.txt: all relations and corresponding ids, one per line.
vec.txt: the pre-train word embedding file.
entity2id.txt: the entity to id file.

## Source Code

The source code is the AtNRE/src/main.py file
Other required files are in the src folder

## Tests

Unit tests have been provided in the tests/ folder

## Benchmarking Datasets

The following datasets have been used: 
	NYT Dataset
	SemEval Dataset
	DDI Dataset

## Evaluation Results (RMSE) 

NYT Dataset

Neural Net | PCNN | RNN |
--- | --- | --- |
AtNRE | 0.67 | 0.73 |
AtNRE/DITK | 0.65 | 0.71 |  

SemEval Dataset

Neural Net | PCNN | RNN |
--- | --- | --- |
AtNRE | N/A | N/A |
AtNRE/DITK | 0.64 | 0.72 |  

DDI Dataset

Neural Net | PCNN | RNN |
--- | --- | --- |
AtNRE | N/A | N/A |
AtNRE/DITK | 0.73 | 0.79 |  


## Helpful demo links

Jupyter Notebook: https://github.com/aru-jo/ditk/blob/develop/extraction/relation/AtNRE/demo/AtNRE.ipynb

Youtube Link: N/A


