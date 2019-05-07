# Location Name Extraction
## Paper Title
- OPA2Vec: Combining formal and informal content of biomedical ontologies to improve similarity-based prediction
## Full Citation
- OPA2Vec: Combining formal and informal content of biomedical ontologies to improve similarity-based prediction -
 Fatima Zohra Smaili, Xin Gao, Robert Hoehndorf. Bioinformatics, 2018
 

## Original Code
https://github.com/bio-ontology-research-group/opa2vec

## Requirements
- Python3
- Groovy (Groovy Version: 2.4.10 JVM: 1.8.0_121) with Grape for dependency management (http://docs.groovy-lang.org/latest/html/documentation/grape.html).
- Perl 5.16.3.



## Description
- The overall task is to convert an ontology into embeddings. In this case the algorithm is specifically designed for biomedical ontology such as Gene Ontology, using protein association and gene ontology the algorithm converts each protein in the association to a list of vector
- The appraoch for the task is as follows:
	- Ontology Processing - Using Hermit reasoner the ontolgy is processed a new axioms are deduced. The deduction is based on subsumption and equality relationship
	- Metadata extraction - From ontology class metadata such as labels and also known as tags are extracted for refernce
	- Corpus Creation : Using above axioms and metadata, a corpus of sentences are created relating protiens to their gene class
	- Running Word2Vec : On the above corpus a pretrained word2vec model pretrained on PubMed Extracts is run
	- Finally the vectors obtained from the word2vec model represents the final embedding of the proteins
## Functions
- The read_dataset function receives list of input file and returns the ontolgy and association files
- The learn_embedding function receives the above files and creates the embedding by follwing all the above steps listed
- The evaluate function evaluates the performance of the embedding, in this case a simple cosine similarity function is created to evaluate semantic simalarity of 2 vectors


## Input and Output
- - **ontology file**            File containing ontology in owl format.
  
 - **association file**         File containing entity-class associations.
Sample association:
```
entrez_943 <http://purl.obolibrary.org/obo/MP_0005673>
```

- Output: A list of embedding
Sample:
```
A0A021WW32 [-0.03606246 -0.01861037 -0.009504    0.04615484  0.03250784  0.03710574
  0.04208963  0.03959762  0.05486397  0.02596879  0.02437145 -0.00500172
  0.01339678 -0.01954783  0.01419695  0.01996015  0.04372361 -0.01296109
  0.02506819 -0.01399953  0.02374094  0.06512783 -0.02932927  0.00186256
  0.00810841  0.0026053  -0.02853323  0.03312564 -0.03519195 -0.04293977
 -0.0339314   0.01872902 -0.04037376  0.03061615  0.02343579  0.02472131
  0.0260462   0.01962205  0.02118931 -0.00392774  0.03743319  0.05130114
 -0.00733691  0.00193934 -0.02811906  0.00457817  0.02694011 -0.03644023
 -0.01123749  0.02816814  0.04620524  0.02853445 -0.01195289  0.01942546
 -0.05365792  0.0087761  -0.00234237 -0.00201961  0.00932485 -0.02874386
  0.03676541  0.01451022 -0.04405008  0.0322992   0.00564483 -0.0028206
 -0.04679357  0.05521384  0.03491683 -0.01005041  0.0036194   0.06403358
  0.01719473  0.01717792  0.01565468 -0.02577724  0.00779381  0.01603061
 -0.0351996  -0.01064034 -0.01807014  0.02586666 -0.00836394 -0.06961402
  0.01992756 -0.00420265  0.01904727 -0.00843538 -0.0065486   0.03248052
 -0.00887935  0.03746689 -0.0192683   0.04993264 -0.02607756  0.00648308
  0.01500176 -0.01860241 -0.01103168 -0.03879162  0.01994324 -0.00518331
 -0.01828747  0.0314309   0.02297185 -0.02643678 -0.00660532  0.0424562
  0.00424282 -0.03789086 -0.01050407 -0.03519899  0.00726575  0.04575294
 -0.00143336  0.02047349 -0.02421691 -0.01151534  0.01968346  0.00117857
  0.03271426 -0.02979166 -0.00672508  0.00583293  0.012269    0.02391515
 -0.00154063 -0.00540428  0.0338997  -0.00657599  0.04808448 -0.03669382
 -0.0289789   0.03364602 -0.00047082  0.03753386 -0.0029364  -0.02864629
  0.02781076  0.00436527  0.04085762 -0.00840923  0.03187585  0.01642676
  0.05293952 -0.01709634  0.01307616  0.02577974  0.03463218 -0.0629032
 -0.02561148  0.01766803 -0.00248356 -0.05010839 -0.014065   -0.00335783
  0.01017054 -0.00358918  0.02078317 -0.0186979   0.02781605  0.05479031
 -0.01000836  0.00838301 -0.00887881 -0.004086   -0.01729655  0.04156226
 -0.00737557 -0.00836478  0.00219193  0.00906456  0.00406466  0.03243787
  0.03747438 -0.00541947  0.02477744 -0.03440936  0.00127687  0.03438043
 -0.00451076  0.01586543 -0.00548162 -0.04289436 -0.07196113 -0.03335271
 -0.03321014  0.01092954  0.0279038   0.00666554 -0.00101996 -0.01093404
 -0.00016811  0.00028596 -0.04079857  0.00621284  0.03085257  0.04827708
 -0.01660185  0.0015438 ]
```

## Evalution
### Benchmark datasets 
- Human and yeast protein instance from Gene Ontology (GO)

### Evaluation metrics
- Cosine Similarity
- AUC 


Benchmark     | Cosine Similarity |   AUC   | 
------------- | ----------------- |-------- |
Gene Ontology |       0.81        |  0.78   |



## Demo
- [Sample notebook](./main_nb.ipynb)
- [Demo Video](https://youtu.be/gnTwA1Cj7J8)