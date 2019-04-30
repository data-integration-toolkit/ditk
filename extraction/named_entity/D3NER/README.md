# Information

D3NER: biomedical named entity recognition using CRF-biLSTM improved with fine-tuned embeddings of various linguistic information

Thanh Hai Dang, Hoang-Quynh Le, Trang M Nguyen, Sinh T Vu, D3NER: biomedical named entity recognition using CRF-biLSTM improved with fine-tuned embeddings of various linguistic information, Bioinformatics, Volume 34, Issue 20, 15 October 2018, Pages 3539–3546, https://doi.org/10.1093/bioinformatics/bty356

# Installation

1) Install pip requirements


2) Download spacy module
python -m spacy download en_core_web_md

3) Clone D3NER and uncompile it using uncompyle6

4) Add an ```__init__.py``` to to the top level package

5) Rename all imports and strings to refer internally to D3NER as a package

6) Refer to directions on installation here https://github.com/trangnm58/D3NER 


# Testing

Input format: Titles and abstracts

Output format: DITK output

Accepts Conll2003, Ditk and BioCreative (.tsv & .txt) format

# Training
Input format: Titles and abstracts, annotations

Output format: DITK output

Accepts Conll2003, Ditk and BioCreative (.tsv & .txt) format

# Task, Method, Model

This is designed for chemical named entity recognition (NER) for the biocreative V4 competition.

From the abstract: "ong Short-Term Memory (LSTM) networks have recently been employed in various biomedical named entity recognition (NER) models with great success. They, however, often did not take advantages of all useful linguistic information and still have many aspects to be further improved for better performance."

# Benchmark datasets

| | F1 | Precision | Recall |
|---|---|---|---|
|CEMP | 73% | 70% | 76% | 
|ChemDNER | 6.7% | 5.4% | 8% |
|conll2003 | 0% | 0% | 0% |

# Video
https://youtu.be/9guuVLIg_Ww

# Notebook
DITKD3nerNotebook.ipynb
