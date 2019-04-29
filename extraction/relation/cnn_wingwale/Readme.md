# Relation classification via convolutional deep neural network

### Refactored by Wing Wa Leung (wingwale@usc.edu)

## Title of the paper
[Relation Classification via Convolutional Deep Neural Network](https://www.aclweb.org/anthology/C14-1220)

## Full citation
Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao. Relation Classification via Convolutional Deep Neural Network.  The 25th International Conference on Computational Linguistics (COLING), Dublin, Ireland, 2014.

## Overview


## Input/Output format for the model
### Inputs
* Sentence
* Two named entities

### Output
* Relation type

## Benchmark datasets
1. DDI2013
    * 5 relation types
    * Training set: 27792 samples
    * Testing set: 5716 samples

2. SemEval2010 Task8
    * 19 relation types
    * Training set: 8000 samples
    * Testing set: 2717 samples

3. NYT dataset
    * 25 relation types
    * Training set: 235275 samples
    * Testing set: 388 samples

## Evaluation metrics and results
### Evaluation metrics
Marco-averaged
* Precision
* Recall
* F1-score


### Evaluation results
![evaluation results](evaluation.png)

## [Link to Jupyter notebook](cnn_notebook.ipynb)

## [Link to Youtube video]()