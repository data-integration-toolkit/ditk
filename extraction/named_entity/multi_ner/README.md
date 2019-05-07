# multi_ner
- Robust Multilingual Named Entity Recognition with Shallow Semi-Supervised Features
- ```
@inproceedings{
  title = {Robust Multilingual Named Entity Recognition with Shallow Semi-Supervised Features}, 
  author = {Rodrigo Agerri, German Rigau}, 
  booktitle = {	Artificial Intelligence, 238, 63-82}, 
  year = 2016, 
}
```

## Required
- Use TensorFlow >=1.6
- Python 3.6

## Description
- Using multiple forms of local features can lead to better NER for multiple languages.

## Input and Output
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
| CoNLL-2003-en | 0.9156 | 0.9217 | 0.9187 |  

## Demo
- Notebook found in repo 
- [Link to the video on Youtube](https://youtu.be/WTYvgFLfJSA)
