# Bayesian Hybrid Matrix Factorisation for Data Integration
This project contains an implementation of the Bayesian hybrid matrix factorisation model presented in the paper 

[**Bayesian hybrid matrix factorisation for data integration**](https://arxiv.org/abs/1704.04962), Thomas Brouwer, Pietro Lio', published at the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017).

<img src="./images/in_out_of_matrix_and_mf_mtf_and_multiple_mf_mtf.png" width="65%"/> <img src="./images/hmf_overview.png" width="33%"/> 

#### Original Code

https://github.com/ThomasBrouwer/HMF

#### Input and Output

Input:
This model takes as input a dataset which may or may not have missing values
If no missing values, it introduces missingness into the data

Output:
The model outputs a predicted dataset with predicted values for the missing data

#### Description

Task: Apply Non Negative Matrix Factorization for Data Imputation

Approach:
• Decompose a given matrix (dataset) into two smaller matrices (latent factors), so that their product approximates the original matrix
• This method extracts hidden structure in the data, and allows the prediction of missing values

#### Evaluation

The benchmark datasets used for this project were:

1. UCI Breast Cancer Dataset - http://archive.ics.uci.edu/ml/datasets/breast+cancer
2. UCI Spam Dataset - http://archive.ics.uci.edu/ml/datasets/spambase
3. UCI Letters Dataset - http://archive.ics.uci.edu/ml/datasets/Letter+Recognition

The evaluation metric used was Root Mean Square Error (RMSE)

Results:

. UCI Breast Cancer Dataset - RMSE: 0.4950865997
2. UCI Spam Dataset - RMSE: 0.128437365988
3. UCI Letters Dataset - RMSE: 1.13686353267



## To Run 
Run the main.py file

(For variable input data, provide the path to the datasets folder)
(For variable fileName, provide the particular dataset you would like to run)

## Citation
> Thomas Brouwer and Pietro Lió (2017). Bayesian Hybrid Matrix Factorisation for Data Integration. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017).
```
@inproceedings{Brouwer2017a,
	author = {Brouwer, Thomas and Li\'{o}, Pietro},
	booktitle = {Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)},
	title = {{Bayesian Hybrid Matrix Factorisation for Data Integration}},
	year = {2017}
}
```
