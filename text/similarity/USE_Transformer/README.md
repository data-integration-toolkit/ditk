Reference
========

Universal Sentence Encoder for English
-------------------

D.Cer, Y.Yang, S.Kong, N.Hua, N.Limtiaco, R.S.John, N.Constant, M.G.Cespedes, S.Yuan, C.ar, Y.Sung, B.S
trope, R.Kurzweil, “_Universal Sentence Encoder for English_”, In Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP 2018), Nov 2018,
pp. 169–174   
http://aclweb.org/anthology/D18-2029

code by author: https://tfhub.dev/google/universal-sentence-encoder-large/3

Toolkit description
==================
This is a toolkit to predict sentences pair's semantic similarity scores, the main idea of the toolkit is first to use transformer encoder to encode two sentences into two embeddings, 
and then calculate two sentences embeddings' similarity by angular distance or cosine similarity. 

sample work flow: read_dataset,load_module,prediction,evalution.
(because the paper's author packed the transformer module in tensorflow hub, we can just download module and use it.So,there is no training details.)
<div align = center>
The architecture of transformer:  
<img src="https://github.com/JoeyJoey/ditk/blob/develop/text/similarity/USE_Transformer/picture/transformer.jpg" width="450" height="600" alt="Transformer"/>
</div>  

Input for prediction: sentences_1,sentences_2
(sentences_1,sentences_2 should be a list, sentence_1[i],sentences_2[i] is a sentence pair)
Output for prediction: similarity scores
(also a list,similarity scores[i] for sentence_1[i] and sentences_2[i])

Evaluation metrics:  
------------------
&emsp;Pearson Correlation Coefficient
 
Evaluation dataset and result:
------------------------------

| Dataset       | pearson (cosine) | pearson( angular) |
| ------------- | -------------    | -------------     |
| SemEval 2017 Task1 track5 | 0.83 | 0.84              |
| SemEval 2014 image        | 0.86 |0.85               |
| SICK test Dataset         | 0.82 | 0.80              |
| STS-benchmark test        | 0.76 | 0.77              |

How to run the code
==================
 the code' interpreter is python 3.6
  1. install packages  
  &emsp;pip3 install requirement.txt
  2. you may download transformer module in tensorflow hub and save it in local directory
  ```# Create a folder for the TF hub module.
$ mkdir /model/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC /model/moduleA
```   
&emsp;3. main.py is a demo to predict sentence pairs similarity score   
&emsp;4. use_transformer_demo.ipynb is a demo that contains prediction and evaluation  

video demo
==========
https://youtu.be/1v9nQXxPVNY

Jupyter notebook
================
use_transformer_demo.ipynb












  


