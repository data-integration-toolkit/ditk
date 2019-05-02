# byteNER

Refactored by Xiangci Li, xiangcil@usc.edu

Video available [here](https://youtu.be/NXu--QkAfLw)


## Description
A named entity recognition tool designed for biomedical NER proposed by [*Sheng, E., & Natarajan, P. (2018). A Byte-sized Approach to Named Entity Recognition. arXiv preprint arXiv:1809.08386*](https://arxiv.org/abs/1809.08386
). Original code available [here](https://github.com/ewsheng/byteNER).

### Idea
Using word tokens and subword tokens as features and using CNN+bidirectional LSTM + CRF to tag entities with BIOES schema for each subword

## Requirements
* Python 2.7
* Keras 
* Tensorflow
* scikit-learn

## Usage
* Follow `byteNER.ipynb` for usage.
* You must split the dataset in the common format defined by ditk.extraction.named_entity into `byteNER/examples/training.tsv`, `byteNER/examples/development.tsv`, `byteNER/examples/evaluation.tsv` and run the code in `byteNER.ipynb` sequentially.
* All functions read data from files and store results back to files.
* Run `testByteNER.py` for test.