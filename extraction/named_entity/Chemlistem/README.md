#Information

Title: Chemlistem: chemical named entity recognition using recurrent neural networks

Corbett, P., & Boyle, J. (2018). Chemlistem: Chemical named entity recognition using recurrent neural networks. Journal of Cheminformatics, 10(1). doi:10.1186/s13321-018-0313-8

#Installation

Because chemlistem is packaged as a pip package, only need to install that package and tensorflow. 

#Testing

Input format: Titles and abstracts

Output format: DITK output

Accepts Conll2003, Ditk and BioCreative (.tsv & .txt) format

#Training
Input format: Titles and abstracts, annotations

Output format: DITK output

Accepts Conll2003, Ditk and BioCreative (.tsv & .txt) format

#Task, Method, Model

This is designed for chemical named entity recognition (NER) for the biocreative V5 competition.

From the paper abstract: "The first system translates the traditional CRF-based idioms into a deep learning framework, using rich per-token features and neural word embeddings, and producing a sequence of tags using bidirectional long short term memory (LSTM) networks—a type of recurrent neural net. The second system eschews the rich feature set—and even tokenisation—in favour of character labelling using neural character embeddings and multiple LSTM layers. The third system is an ensemble that combines the results of the first two systems."

#Benchmark datasets
	F1	Precision	Recall
CEMP: 	80%	73%		80%
ChemDNER:88%	90%		87%
conll2003:0%	0%		0%

#Video
https://youtu.be/bVtRWtMK7kI

#Notebook
DITKChemListemNotebook.ipynb
