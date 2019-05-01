Reference
========

Siamese Recurrent Architectures for Learning Sentence Similarity
-------------------

J.Mueller, A.Thyagarajan. “_Siamese Recurrent Architectures for Learning Sentence Similarity_”, In Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016). pp. 2786-2792

paper's url: http://aclweb.org/anthology/D18-2029

code by author: https://github.com/aditya1503/Siamese-LSTM

Toolkit description
==================
This is a toolkit to predict two sentence's semantic similarity score, the main step of the toolkit can be included as 4 steps:  
1.Use gensim word2vec model via Google-news-vectos to translate words into word vectors
2.Use an LSTM to read in word-vectors representing each input sentence 
3.employs its final hidden state as a vector representation for each sentence
4.the similarity between these representations is used as a predictor of semantic similarity.

The architecture of the model:  
<img src="https://github.com/JoeyJoey/ditk/blob/develop/text/similarity/Siamese_LSTM/picture/Siamese_LSTM.jpg" width="450" height="600" alt="Siamese_LSTM"/>     
<br>

sample work flow of the toolkit: read_dataset,load_model (or train model),prediction,evaluation.

Input for prediction: sentence_1,sentence_2

Output for prediction: similarity score of two sentences

Evaluation metrics:  
------------------
&emsp;Pearson Correlation Coefficient
 
Evaluation dataset and result:
------------------------------

| Dataset       | pearson (cosine) |
| ------------- | -------------    |
| SemEval 2017 Task1 track5 | 0.70 | 
| SemEval 2014              | 0.71 |
| SICK test Dataset         | 0.71 |


How to run the code
==================
 the code' interpreter is python 3.6
  1. install packages  
  &emsp;pip3 install -r requirement.txt
  2. Please download GoogleNews-vectors-negative300.bin and put it under Siamese_LSTM directory.
  
  3. You can use trained model bestsem.p under model directory, by load_model(filepath) method, see main.py
  4. If you want to train model, please use train(traindata,maxepochs) method , see src/train_demo.py. please make sure your train_data'sformat are as follows:
  [[s1,s2,score],[s1,s2,score],...] a list consists of several lists which contains sentence1,sentence2,similarity_score
  5. you can save your model after training by save_model(directory) method. It use pickle.dump() method to save model into file.

video demo
==========
https://youtu.be/CgiC_mwLDYM

Jupyter notebook
================
src/simple_demo.ipynb















  


