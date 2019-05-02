# Named Entity Recognition and Part of Speech Tagging with S-LSTM
## Title of the paper: Named Entity Recognition and Part of Speech Tagging with S-LSTM

## Instructions to run the code:
[Instructions](https://drive.google.com/drive/folders/1ZVmuEAJ31yMNkYMUoyWG1hf2HZa6FJyn)

## Full citation :
```
@article{zhang2018slstm,
title={Sentence-State LSTM for Text Representation},
author={Zhang, Yue and Liu, Qi and Song, Linfeng},
booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)}, year={2018}
}
```
## Original Code :
https://github.com/leuchine/S-LSTM/tree/master/sequence_tagging

## Description:
An alternative LSTM structure for encoding text, which consists of a parallel state for each word. Re-current steps are used to perform local and global information exchange between words simultaneously, rather than incremental reading of a sequence of words.

## Input and Output
Input Format: [Link](https://github.com/divyasinha801/ditk/blob/develop/extraction/named_entity/slstm/ner_test_input.txt)

Output Format: {entity} {predicted tag} {ground truth}

## Evalution
Benchmark datasets:
1. CoNll
2. Ontonotes

Evaluation metrics and results:
Accuracy : 93.59
F1 : 75
Precision : 75
Recall : 75

## Demo
Link to the Jupyter Notebook: [Link](https://github.com/divyasinha801/ditk/blob/develop/extraction/named_entity/slstm/Sentence%20State%20LSTM%20(S-LSTM)%20for%20Named%20Entity%20Recognition%20(Group%206).ipynb)

Link to the video on Youtube: https://www.youtube.com/watch?v=Yz4NYFkob90&feature=youtu.be
