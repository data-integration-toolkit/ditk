ConEc (Context Encoders), an extension of word2vec
----------------------------------------------------
This is a Python3 implementation of the following paper - 

# Context encoders as a simple but powerful extension of word2vec
Franziska Horn, 
Proceedings of the 2nd Workshop on Representation Learning for NLP, 
Association for Computational Liguistics 2017, Pages 10-14

Installation
------------

pip install -r requirements.txt

This will install all the prerequisite packages	for conec to run successfully

Introduction
------------
The task is to obtain embeddings for Out of Vocabulary (OOV) words and for words having multiple meanings by exploiting the word's local context

INPUTS (For Prediction and Training)
------------------------------------
Word | Sentence | Paragraph | Document

OUTPUTS (For Prediction and Training)
-------------------------------------
- Text Embedding | Average Text Embedding
- Embedding Similarity

Objective
---------
This implementation mitigates the shortcomings of word2vec in that word2vec learns only a single representation (embedding) for a word, but it is possible to have a word which means different things in different context..
For Ex. Washington (US State) vs Washington (Former President).

This code can be used to train, evaluate an use Context Encoders (ConEc), a powerful context-aware extension of word2vec. This can be used to generate text embeddings for a word with multiple meanings or for Out-Of-Vocabulary (OOV) words by using a trained model. 

Architecture Diagram
--------------------
Architecture for the ConEc model

![conec_architecture](https://user-images.githubusercontent.com/10741993/56942674-c4435680-6ad0-11e9-9025-2aee1d854506.JPG)

Implementation
--------------
The code implements the following procedure -
- Use CBOW word2vec model with negative sampling objective
- Train the model on "text8" , "OneBilCorpus" or "conll2003" datasets
- Load the trained model from its pickle dump
- Get the global context matrix for the same dataset
- Adapt the word embeddings of the word2vec model by multiplying it the 
  word's average context vectors (CVs)
- A word has global CV and local CV
- Choice of alpha in the equation mentioned in the paper determines the emphasis on the word's local context

![conec_equation](https://user-images.githubusercontent.com/10741993/56942816-78dd7800-6ad1-11e9-981f-56b5c8c1734d.JPG)

- Renormalize the result, so that the resulting embeddings have unit length again

Download the conec folder from this repository and import the conec class into your script

DATASETS 
---------
For training :
1) Google Analogy Dataset
   
   Download from - https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt
2) One Billion Words Corpus
   
   Download from - https://code.google.com/archive/p/1-billion-word-language-modeling-benchmark/
3) CoNLL 2003, NER Task
   
   Download from - https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en
   
   CoNLL evaluation script - https://www.clips.uantwerpen.be/conll2003/ner/
4) SICK 2014
   
   Download from - https://github.com/brmson/dataset-sts/tree/master/data/sts/sick2014	 	
5) SemEval 2014
   
   Download from - https://github.com/brmson/dataset-sts/tree/master/data/sts/semeval-sts/2014

RESULTS
-------
Table showing the Evaluation metrics for Benchmarks used along with the results

![conec_evaluation](https://user-images.githubusercontent.com/10741993/56942708-e937c980-6ad0-11e9-941a-e9af35007b43.JPG)

GRAPHS
------
Graphs plotting Accuracies anf F1 scores for Google Analogy Dataset and CoNLL 2003 NER Task, respectively

![conec_ner](https://user-images.githubusercontent.com/10741993/56942774-3a47bd80-6ad1-11e9-86e7-13334add228b.png)

![conec_analogy](https://user-images.githubusercontent.com/10741993/56942783-43d12580-6ad1-11e9-8de9-7c7f06a89b09.png)

Video Demonstration
-------------------


Jupyter Notebook Link
---------------------
https://github.com/bjainvarsha/ditk/blob/develop/text/embedding/conec/conec.ipynb


