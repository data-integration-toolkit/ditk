# HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding
## Title of the paper: HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding
## Full citation :
```
InProceedings{D18-1225,
  author = 	"Dasgupta, Shib Sankar
		and Ray, Swayambhu Nath
		and Talukdar, Partha",
  title = 	"HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"2001--2011",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1225"
}
```
## Original Code :
https://github.com/malllabiisc/HyTE

## Description:
A temporally aware KG embedding method which leverages validity of triples in a KG. It incorporates time in the entity-relation space by stitching each timestamp with a corresponding hyperplane. HyTE not only performs KG inference using temporal guidance, but also predicts temporal scopes for relational facts with missing time annotations. 
reference : [Link](https://github.com/malllabiisc/HyTE)

## Input and Output
Input format : <entity> <relation> <entity> <timestamp> <timestamp>
Output : valid triples and embeddings

## Evalution
Benchmark datasets:
1. YAGO
2. Wikidata

Evaluation metrics and results:
MR:
Test tail rank : 3.769230769230769
Test_head rank : 12.0

Hit@10:
Test_tail HIT@10 : 92.3076923076923
Test_head HIT@10 : 92.3076923076923

## Demo
Link to the Jupyter Notebook:

Link to the video on Youtube: https://www.youtube.com/watch?v=Yz4NYFkob90&feature=youtu.be
