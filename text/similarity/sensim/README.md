## UdL at SemEval-2017 Task 1: Semantic Textual Similarity Estimation of English Sentence Pairs Using Regression Model over Pairwise Features

Given two sentences returns the similarity score between them as a value between 0 and 1

Implemented the paper  [UdL at SemEval-2017 Task 1: Semantic Textual Similarity Estimation of English Sentence Pairs Using Regression Model over 
Pairwise Features](https://www.aclweb.org/anthology/S17-2013)


Input  : 2 sentences

Output :  Similarity Score

Developed in python3

## Approach

To compute the similarity we use 5 features which are

 1) Average of w2v Cosine Similarity of all the PoS tag pairs
 2) Average of w2v Cosine Similarity of all the NE tag pairs
 3) Cosine similarity of the sentence BoW vector pair
 4) Absolute difference of the number summation
 5) Absolute difference of the number of characters
 
 This figure shows the importance of each feature in predicting a similarity score
 
 ![alt text](https://github.com/Sanjithae/sensim/blob/master/Figure1.PNG)
 
 After obtaining the features we apply Random Forest Ensemble Learning to predict the final similarity score.
 
 ## To run the program
 
 The input to the program should be a csv file which consists of 3 columns sentence1,sentence2,similarity score
 
 python main.py
 
 
 ## Evaluation Metrics
 
 Pearson Correlation Coefficient.
 
 ## Benchmark Datasets 
 
 1) SICK 2014
 
 2) SemEval 2017
 
 3) SemEval 2014
 
 ## To make this model to run for other datasets
 
 If you want to use the trained model for a new dataset
 
 1) Create a folder with the name of the dataset in the data folder
 2) Change the filepath in the main.py and sensim.py
 3) Run main.py
 
 If you want to use the trained model for new sentences 
 
 1) In the sensim.py in the evaluate function use the trained model for predicting by calling the load_model() and comment the code which is responsible for creating a new model.
 2) Run main.py
 

## Results

SemEval 2017 = 39.46

SemEval 2014 = 44.89

SICK         = 42.54

## Original Code

Code provided by the author :

https://github.com/natsheh/sensim

Actual code used in this repository :

https://github.com/Sanjithae/sensim


## Jupyter Notebook

You can see the execution of each function [here.](https://github.com/Sanjithae/sensim/blob/master/sentence_similarity.ipynb)

## Youtube Video

Explanation of the code [here.](https://youtu.be/X_8rV55Kxyc)



## Citation
Al-Natsheh, Hussein T.  and  Martinet, Lucie  and  Muhlenbach, Fabrice  and  ZIGHED, Djamel Abdelkader,
UdL at SemEval-2017 Task 1: Semantic Textual Similarity Estimation of English Sentence Pairs Using Regression Model over 
Pairwise Features,Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017),August,2017,Vancouver, Canada,Association for Computational Linguistics,115--119.
(http://www.aclweb.org/anthology/S17-2013)
