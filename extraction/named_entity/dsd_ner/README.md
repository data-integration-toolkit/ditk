# DSD_Ner
- Learning Named Entity Tagger using Domain-Specific Dictionary
- ```
@inproceedings{shang2018learning,
  title = {Learning Named Entity Tagger using Domain-Specific Dictionary}, 
  author = {Shang, Jingbo and Liu, Liyuan and Ren, Xiang and Gu, Xiaotao and Ren, Teng and Han, Jiawei}, 
  booktitle = {EMNLP}, 
  year = 2018, 
}
```

## Reqs
keras==2.1.2
sklearn
nltk
tensorflow==1.13.1
numpy


## Description
- Use domain specific dictionaries and/or word embeddings to create more robust and accurate predictions for NER in domains with fewer supervised data sets.

## Input and Output
- Input: Sentence (Non-Tokenized) Character level features, Word embeddings
- Training input:
	- Matches ditk NER team decision.
	- 12 columns as below. ConLL 2003 + ConLL 2012 format
	```
Yes UH (TOP(S(INTJ*) O bc/cnn/00/cnn_0003 0 0 - - - Linda_Hamilton * -
they PRP (NP*) O bc/cnn/00/cnn_0003 0 1 - - - Linda_Hamilton * (15)
did VBD (VP*) O bc/cnn/00/cnn_0003 0 2 do 01 - Linda_Hamilton (V*) -
/. . *)) O bc/cnn/00/cnn_0003 0 3 - - - Linda_Hamilton * -
```
## Evalution
- Benchmark Datasets
    - CoNLL-2003
- Evaluation metrics
    - Precision
    - Recall
    - F-1 score
- Results (Test set)

| Dataset | Precision | Recall | F-1 score | 
| :--- | :---: | :---: | :---: | 
| CoNLL-2003-en | 0.8927 | 0.9118 | 0.9021 |  

## Demo
- Notebook found in repo.
- [Link to the video on Youtube](https://youtu.be/coFmCHVb-BI)
