# Midas
Lovedeep Gondara and Ke Wang, **"MIDA: Multiple Imputation using Denoising Autoencoders,"** Pacific-Asia Conference on Knowledge Discovery and Data Mining, 2018.

Link to paper - https://arxiv.org/pdf/1705.02737.pdf

## Original Code
Github URL of the original code - https://github.com/Oracen/MIDAS/

## Description of the problem
- Missing data is an important issue, even small proportions of missing data can adversely impact the quality of learning process, leading   to biased inference. This paper brings in the concept of **Multiple Imputation**. All previous standalone methods shared a common drawback, imputing a single value for one missing observation, which is then treated as the gold standard, same as the observed data in any  subsequent analysis. This implicitly assumes that imputation model is perfect and fails to account for error/uncertainty in the imputation process. This is overcome by replacing each missing value with several slightly different imputed values, refecting our uncertainty about the imputation process. This approach is called * *multiple imputation.* *

## Architecture: 

![Architecture](https://github.com/karishma-chadha/ditk/blob/develop/data_cleaning/midas/readme-images/midas_arch.png)

- We start with an initial n dimensional input ([0,1]) and feed it into the encoder. Stochastic corruption is induced by keeping the input  dropout ratio as 0.5. Then at each successive hidden layer, we add theta nodes, increasing the dimensionality to n+theta. This mapping of our input data to a higher dimensional subspace creates representations capable of adding lateral connections, aiding in data recovery. Decoder symmetrically scales it back to original dimensions and tries to reconstruct the input. In this process, the algorithm is tuned to predict the missing values.
- This technique can work with pre-existing missing values, without the need to replace those values with mean or other similar metric.

## Working of Code
- We **input** gold numeric dataset in csv format. The code then, introduces missingness in the dataset(20%) and the algorithm is run on this missing data.
- We get **output** as complete dataset in a newly created csv file.
- We provide gold dataset in order to evaluate the **performance of this algorithm by rmse value.**

## Input and Output for prediction and training
- Input  : path to csv file having gold numeric relation.
- Output : newly created csv file with predicted values in the relation.

## Evalution
- Benchmark datasets :
  UCI Repository Datasets
 (complete, numerical) 
- Letter Recognition - http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
- Breast Cancer (Diagnostic) - http://archive.ics.uci.edu/ml/datasets/breast+cancer
- Spam-base - http://archive.ics.uci.edu/ml/datasets/spambase

- Evaluation metrics:
  Root Mean Square Error (RMSE) 
  
 - Evaluation results
 
![Evaluation Results](https://github.com/karishma-chadha/ditk/blob/develop/data_cleaning/midas/readme-images/midas_eval_results.PNG) 

## Demo
- Link to the Jupyter Notebook 
- Link to the video on Youtube
