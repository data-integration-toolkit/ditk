# Data Integration Took Kit for CSIC548 Final Project

## Title of the Paper: 
- Cross-type biomedical named entity recognition with deep multi-task learning

## Full Citation: 
- Xuan Wang, Yu Zhang, Xiang Ren, Yuhao Zhang, Marinka Zitnik, Jingbo Shang, Curtis Langlotz, Jiawei Han; Cross-type biomedical named entity recognition with deep multi-task learning, Bioinformatics, bty869. 

## Overview: 
The proposed method used multi-task mechanism to build a general moodel for biomdeical NER. Each dataset has its own model and shares word level, character level weights during the training step.    

## Format for prediction: 
 - input: Document (Each line contains one word)
 - output Document (Each line contains one word and corresponding tag)

## Format for training: each column is separated by single space  
 - First column: word 
 - Second  column: POS tag 
 - Third column: syntactic chunk tag 
 - Fourth column: named entity tag

## Benchmark datasets:
- BC2GM
- BC4CHEMD 

## Evaluation metrics and results (10 epoch):
- F1: 72.88%
- Recall: 77.88%
- Precision: 68.48%

## Links:
- Jupyter Notebook: https://github.com/hsia947/ditk/blob/develop/extraction/named_entity/bio/Multi-BioNER/multibio_main.ipynb
- Youtube: https://youtu.be/fxD_JorwbGA






