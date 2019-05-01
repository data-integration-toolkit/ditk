# Bayesian hybrid matrix factorisation for data integration
This project contains an implementation of the Bayesian hybrid matrix factorisation models presented in the paper [**Bayesian hybrid matrix factorisation for data integration**](https://arxiv.org/abs/1704.04962), published at the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017). We furthermore provide all datasets used (including the preprocessing scripts), and Python scripts for experiments.

<img src="./images/in_out_of_matrix_and_mf_mtf_and_multiple_mf_mtf.png" width="65%"/> <img src="./images/hmf_overview.png" width="33%"/> 

#### Paper abstract
We introduce a novel Bayesian hybrid matrix factorisation model (HMF) for data integration, based on combining multiple matrix factorisation methods, that can be used for in- and out-of-matrix prediction of missing values. The model is very general and can be used to integrate many datasets across different entity types, including repeated experiments, similarity matrices, and very sparse datasets. We apply our method on two biological applications, and extensively compare it to state-of-the-art machine learning and matrix factorisation models. For in-matrix predictions on drug sensitivity datasets we obtain consistently better performances than existing methods. This is especially the case when we increase the sparsity of the datasets. Furthermore, we perform out-of-matrix predictions on methylation and gene expression datasets, and obtain the best results on two of the three datasets, especially when the predictivity of datasets is high.

#### Authors
Thomas Brouwer, Pietro Lio'. 
Contact: thomas.a.brouwer@gmail.com.

## Installation 
If you wish to use the matrix factorisation models, or replicate the experiments, follow these steps. Please ensure you have Python 2.7 (3 is currently not supported). 
1. Clone the project to your computer, by running `git clone https://github.com/ThomasBrouwer/HMF.git` in your command line.
2. In your Python script, add the project to your system path using the following lines.  
   
   ``` 
   project_location = "/path/to/folder/containing/project/"
   import sys
   sys.path.append(project_location) 
   ```
   For example, if the path to the project is `/johndoe/projects/HMF/`, use `project_location = /johndoe/projects/`. 
   If you intend to rerun some of the paper's experiments, those scripts automatically add the correct path.
3. You may also need to add an empty file in `/johndoe/projects/` called `__init__.py`.
4. You can now import the models in your code, e.g.
```
from HMF.code.models.hmf_Gibbs import HMF_Gibbs
import numpy
R1, R2 = numpy.ones((4,3)), numpy.ones((4,3))
M1, M2 = numpy.ones((4,3)), numpy.ones((4,3))
C, D, R = [], [], [(R1,M1,'row_entities','col_entities',1.), (R2,M2,'row_entities','col_entities',1.)]
K = { 'row_entities': 2, 'col_entities': 2 }
model = HMF_Gibbs(R=R,C=C,D=D,K=K)
model.initialise()
model.run(iterations=10)
model.predict_Rn(n=0,M_pred=M1,burn_in=5,thinning=1)
```

## Examples
A good example of the hybrid matrix factorisation model running on toy data can be found in this [convergence experiment](./toy_experiments/convergence/hmf_gibbs_all.py), which uses a combination of main datasets, feature datasets, and similarity datasets. 

## Citation
If this project was useful for your research, please consider citing our [paper](https://arxiv.org/abs/1704.04962).
> Thomas Brouwer and Pietro Lió (2017). Bayesian Hybrid Matrix Factorisation for Data Integration. Proceedings of the 20th International Conference on Arti cial Intelligence and Statistics (AISTATS 2017).
```
@inproceedings{Brouwer2017a,
	author = {Brouwer, Thomas and Li\'{o}, Pietro},
	booktitle = {Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)},
	title = {{Bayesian Hybrid Matrix Factorisation for Data Integration}},
	year = {2017}
}
```

## Project structure
<details>
<summary>Click here to find a description of the different folders and files available in this repository.</summary>

<br>

### /code/
Code for the HMF model, as well as other matrix factorisation models, helper methods, and more.

**/Gibbs/**: Folder containing the general matrix factorisation Gibbs sampling updates, draws, and initialisations. This code does most of the heavy lifting.
- **draws_Gibbs.py** - Code for drawing new values for matrix factorisation models using Gibbs sampling (given the parameter values).
- **init_Gibbs.py** - Code implementing the different initialisation approaches for the Gibbs sampling parameters (random, expectation, K-means, least squares).
- **updates_Gibbs.py** - Code for computing the Gibbs sampling posterior parameter updates.

**/models/**: Classes that implement the actual matrix factorisation models - NMF, NMTF, BNMF, BNMTF, HMF. These use the code in /Gibbs/ extensively.

**/distributions/**: Python classes that handle computing the mean and variance, and drawing samples, from the probability distributions used in the models.

**/kmeans/**: Implementation of K-means clustering when the matrix is only partially observed. From [my other Github project](https://github.com/ThomasBrouwer/kmeans_missing).

**/kernels/**: Classes for computing, storing, and loading, similarity kernels (Jaccard and Gaussian).

**/statistics/**: Python code for computing the prediction errors of a list or matrix of values, compared to the observed values (MSE, R^2, Rp).

**/model_selection/**: Classes that help with heuristic model selection of the BNMF and BNMTF models (using line search, grid search, or greedy search - see my [NIPS workshop paper](http://arxiv.org/abs/1610.08127)).

**/cross_validation/**: Python scripts for performing in-matrix cross-validation experiments for non-probabilistic and Bayesian matrix factorisation models, and HMF. Some scripts also allow you to do model selection (either nested cross-validation, or using heuristics from NIPS workshop paper).

**/generate_mask/**: Methods for helping generate observation matrices (1 indicating observed entries, 0 indicating unobserved).

**/generate_toy/**: Method for generating toy/synthetic data for the HMF model.

### /drug_sensitivity/
Data and scripts for drug sensitivity data integration experiments. Helper methods for loading in the datasets are provided in load_dataset.py.

**/data/**: Data and preprocessing scripts for drug sensitivity application (GDSC, CTRP, CCLE IC50, CCLE EC50). See description.txt for more details on preprocessing steps.

**/cross_validation/**: In-matrix cross-validation scripts and results for the different methods, for in-matrix predictions. Used in main paper.

**/varying_sparsity/**: Experiments measuring the performance of the matrix factorisation methods (NMF, NMTF, BNMF, BNMTF, HMF D-MF, HMF D-MTF) as the sparsity of the GDSC and CTRP drug sensitivity datasets increases. Used in main paper.

**/runtime/**: Measurements of the runtime performance of the different matrix factorisation models (HMF D-MF, HMF D-MTF, NMF, NMTF BNMF, BNMTF) on the drug sensitivity datasets. Used in supplementary materials.

**/varying_init/**: Exploration of different initialisation methods for HMF D-MF and HMF D-MTF. Used in supplementary materials.

**/varying_K/**: Exploration of effectiveness of ARD, by trying the HMF D-MF and HMF D-MTF models with different values of K, on the four drug sensitivity datasets. Used in supplementary materials.

**/varying_negativity/**: Exploration of the trade-offs between the different negativity constraints (nonnegative, semi-nonnegative, real-valued), on the drug sensitivity datasets. Used in supplementary materials.

**/varying_negativity_sparsity/**: Exploration of the trade-offs between the different negativity constraints (nonnegative, semi-nonnegative, real-valued), on the CTRP drug sensitivity dataset, when the sparsity of the datasets increases. Used in supplementary materials.

**/varying_factorisation/**: Exploration of the trade-offs between the different factorisation combinations (all matrix factorisation, all matrix tri-factorisation, and everything in between), on the drug sensitivity datasets. Used in supplementary materials.

**/bicluster_analysis/**: Analysis of biclusters of running HMF D-MTF on the four drug sensitivity datasets. Very incomplete; not used in paper.

### /methylation/
Data and code for gene expression and methylation data integration experiments. Helper methods for loading in the datasets are provided in load_methylation.py.

**/data/**: Data and preprocessing scripts.
- **/data_plots/** - Visualisation of the data sources (raw, standardised, and similarity kernels). Created by plot_distributions.py.
- **/gene_classes/** - Gene classes for the genes in the methylation data, based on GO terms. See description.txt. Not used in paper.
- **intogen-BRCA-drivers-data.geneid** - list of Entrez Gene ID's for the genes (in order of other datasets).
- **matched_sample_label** - list of sample names, and whether they are tumour or healthy (in order of other datasets).
- **matched_expression** - gene expression data (first row is sample names, first row is gene ids).
- **matched_methylation_geneBody** - gene body methylation data (first row is sample names, first row is gene ids).
- **matched_methylation_genePromoter** - promoter-region methylation data (first row is sample names, first row is gene ids).
- **compute_correlation_datasets.py** - Computes the correlations of the three datasets. Used in supplementary materials.
- **construct_similarity_kernels_genes.py**, **construct_similarity_kernels_samples.py** - Scripts that computes Gaussian similarity kernels for the three datasets, resulting in files: kernel_ge_std_genes, kernel_ge_std_samples, kernel_gm_std_genes, kernel_gm_std_samples, kernel_pm_std_genes, kernel_pm_std_samples.

**/convergence/**: Script for running HMF D-MTF on the gene expression and promoter-region methylation data, to check convergence.

**/cross_validation/**: Cross-validation scripts and results for the different methods, for out-of-matrix predictions. Used in main paper.

**/varying_K/**: Exploration of effectiveness of ARD, by trying the HMF D-MF and HMF D-MTF models with different values of K, on the methylation and gene expression data. Used in supplementary materials.

**/varying_negativity/**: Exploration of the trade-offs between the different negativity constraints (nonnegative, semi-nonnegative, real-valued), on the methylation and gene expression data. Used in supplementary materials.

**/varying_factorisation/**: Exploration of the trade-offs between the different factorisation combinations (all matrix factorisation, all matrix tri-factorisation, and everything in between), on the methylation and gene expression data. Used in supplementary materials.

**/bicluster_analysis/**: Analysis of biclusters of running HMF D-MTF on the gene expression and methylation datasets. Not used in paper.

### /tests/
py.test unit tests for a large portion of the code and classes in **/code/**. To run the tests, simply `cd` into the /tests/ folder, and run `pytest` in the command line.

### /toy_experiments/
Very brief tests on the toy dataset, mainly for checking convergence of the model.

### /images/
The images at the top of this README.

</br>
</details>
