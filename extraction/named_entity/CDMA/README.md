# Objective: Data Integration Took Kit for CSIC548 Final Project

## Title of the Paper: Neural Adaptation Layers for Cross-domain Named Entity Recognition

## Full Citation: Lin, Bill Yuchen and Wei Lu. “Neural Adaptation Layers for Cross-domain Named Entity Recognition.” EMNLP (2018). 

## Overview: 
The proposed method pretrained model with the source domain dataset, then applied its training weights to the target model for training target domain dataset. The result shows that the model with pretrained weights has better performance than the model without pretrained parameters.    

## Format for prediction: 
 - input: Document (Each line contains one word)
 - output Document (Each line contains one word and corresponding tag)

## Format for training: each column is separated by single space  
 - First column: word 
 - Second  column: POS tag 
 - Third column: syntactic chunk tag 
 - Fourth column: named entity tag

## Benchmark datasets:
- Conll2003 (source dataset)
- Ontonotes5.0 (target dataset)

## Evaluation metrics and results:
- F1: 97.15%
- Recall: 83.66%
- Precision: 84.46%

## Links:
- Jupyter Notebook: 
- Youtube: 






