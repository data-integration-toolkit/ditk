# HDL-CRF
This repo is based on the following paper and Github implementation:

Li, Xusheng, et al. "A Hybrid Deep Learning Framework for Bacterial Named Entity Recognition." 2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2018. 

Paper link:
https://ieeexplore-ieee-org.libproxy1.usc.edu/stamp/stamp.jsp?tp=&arnumber=8621446

Github link:
https://github.com/lixusheng1/HDL-CRF 

## Dependencies
* python2
* numpy
* tensorflow

## Data format
* [word, BIO tag, chunk tag]
* [sample data](./data/test_set.iob)

## Benchmark datasets
* PubMed Articles with keywords "bacteria" and "oral"

## Evaluation metrics and results
* Precision: 86.563
* Recall: 86.761
* F1: 86.662

## Jupyter Notebook
* [demo.ipynb](./demo.ipynb)

## YouTube Video
* https://youtu.be/g4dPFgpG8zM