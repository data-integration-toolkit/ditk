# MINHASH - LSH
This project contains an implementation of the MinHash LSH Algorithm.

[**MinHash LSH - Finding Similar Items**](http://infolab.stanford.edu/~ullman/mmds/ch3a.pdf), Jure Leskovec, Anand Rajaraman, Jeff Ullman, published - 'MINING OF MASSIVE DATASETS'. In that book, Chapter 3: Finding Similar Items : They've given alogorthims such as MinHash to represent items in an efficient way such that similar items are located closer to each other. I furthermore provide all datasets used (including the preprocessing scripts), and Python scripts for experiments.

<img src="./images/in_out_of_matrix_and_mf_mtf_and_multiple_mf_mtf.png" width="65%"/> <img src="./images/hmf_overview.png" width="33%"/> 

#### Original Code

https://github.com/chrisjmccormick/MinHash
#### Input and Output

>>Input:
This model takes as input a pair of sentences, on which the similarity has to be determined.

>>Output:
The model outputs :
1) Embedding of the sentence / MinHash signature
2) Similarity score between the 2 input sentences.

#### Description

>>Task: 
- Reduce the high dimensional features to smaller dimensions while preserving the differentiability.
- Group similar objects into same buckets with high probability.


>>Approach:
-	Hashing based algorithm to identify approximate nearest neighbors
-	Sub-linear complexity is achieved by reducing the number of comparisons needed to find similar items. 
-	A hash function h is Locality Sensitive if for given two points
a, b in a high dimensional feature space
•	Pr(h(a) == h(b)) is high if a and b are near
•	Pr(h(a) == h(b)) is low if a and b are far
•	Time complexity to identify close objects is sub-linear
-	Generate, say, 10 random hash functions. Take the first hash function, and apply it to all of the shingle values in a document.
-	 Find the minimum hash value produced and use it as the first component of the MinHash signature.
-	Continue this with all the 10 hash functions. So if we have 10 random hash functions, we’ll get a MinHash signature with 10 values.
-	Compare the documents by counting the number of signature components in which they match

#### Implementation
1.	Load the sentences or documents on which we want to compute approximate similarity
2.	Generate a MinHash object with number of permutations(higher the number  higher the accuracy) and a Hash function ( default : SHA1 )
3.	Update the MinHash signature with the given sentence/document
4.	Compute Jaccard Similarity between the signatures.



#### Evaluation

The benchmark datasets used for this project were:

1. SICK 2014 (SemEval 2014, Task 1)
 - http://alt.qcri.org/semeval2014/task1/
2. SemEval 2014 Task 10
 - http://alt.qcri.org/semeval2014/task10/
3. SemEval 2017 Task 1
 - http://alt.qcri.org/semeval2017/task1/

>>EVALUATION METRICS : 
- Pearson Correlation Coefficient
- Spearman Correlation Coefficient
>> RESULTS :

| DATASET       | PEARSON         | SPEARMAN|
| ------------- | -------------    | -------------     |
| SemEval 2017 Task1  | 0.39 | 0.39              |
| SemEval 2014 image        | 0.44 |0.46               |
| SICK test Dataset         | 0.60 | 0.64              |



## To Run 

1) Download the module code repository -- ditk/text/similarity/minHash
2) Ensure that the main.py, MinHash.py, datasets are in the same folder, if not give the absolute paths accordingly.
3) Run main.py
