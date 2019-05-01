# Ned_with_web_links
- Entity Disambiguation with Web Links
- Andrew Chisholm and Ben Hachey. Published in Transactions of the Association for Computational Linguistics, 2015. <https://aclweb.org/anthology/Q15-1011>

## Original Code
<https://github.com/wikilinks/nel>

## Description
- Entity disambiguation with Wikipedia relies on structured information from redirect pages, article text, inter-article links, and categories. We explore whether web links can replace a curated encyclopaedia, obtaining entity prior, name, context, and coherence models from a corpus of web pages with links to Wiki-pedia. Experiments compare web link models to Wikipedia models on well-known CoNLL and TAC data sets. 
- More detail there: [model detail](https://nel.readthedocs.io/en/latest/)

## Input and Output
- Input: Articles, a list of words and its coresponding name entity tag
- Output: Entity url

## Evalution
- CoNLL
- F1(85.2%)

## Demo
- [notebook](conll_train.ipynb) 
- [vedio](https://youtu.be/I5Z6lo-BY4c)
