# RESCAL

### Refactored by Wing Wa Leung (wingwale@usc.edu)

## Title of the paper
[A Three-Way Model for Collective Learning on Multi-Relational Data](http://www.icml-2011.org/papers/438_icmlpaper.pdf
)

## Full citation
    Maximilian Nickel, Volker Tresp and Hans-Peter-Kriegel. A Three-Way Model for Collective Learning on Multi-Relational Data. Proceedings of the 28th International Conference on Machine Learning, Bellevue, USA, 2011.

## Original Code
[Python module to compute the RESCAL tensor factorization](https://github.com/mnick/rescal.py)

## Overview
* Rescal factorizes each slice of tensor X into 2 matrices A and R<sub>k</sub> using Alternating Least Square (ALS) approach
* A tensor entry X<sub>ijk</sub> = 1 denotes that there exists a relation (i-th entity, k-th predicate, j-th entity)

![rescal](rescal.png)

* Matrix A contains the latent component representation of the entities in the domain
* Matrix R<sub>k</sub> models the interactions of the latent components in the k-th predicate

## Inputs and Outputs
### Inputs
* Tensor model X in MATLAB binary data format (.mat)
    * Shape of X is n * n * m
* Rank r

### Outputs
* For each frontal slice of the tensor X (X<sub>k</sub>):
    * Matrix A (n * r)
    * Matrix R<sub>k</sub> (r * r)
* Matrix A and a list R (which contains matrix R<sub>k</sub> for all values of k) are outputted to text files "rescal_output_A.txt" and "rescal_output_R.txt" respectively

## Evalution
### Benchmark datasets
1. Alyawarra kinship data
    * Collected by Denham
    * 104 entities
    * 26 relations

2. UMLS dataset from a biomedical ontology
    * Prepared by McCray et al.
    * 135 entities
    * 49 relations
    
3. Human disease-symptoms data resource 
    * Prepared by Zitnik et al.
    * 1578 entities
    * 3 relations

### Evaluation metrics
For both training and testing data:
* Mean of Precision Recall Area Under Curve (PR AUC)
* Standard deviation of PR AUC

### Evaluation results
![evaluation results](evaluation.png)

## [Link to Jupyter notebook](rescal_notebook.ipynb)

## [Link to Youtube video](https://youtu.be/94lWBuzM0XA)

## Remarks
* The test cases are written by myself because papers in our group are not related to the same topic 