# mice: Multivariate Imputation by Chained Equations
Stef van Buuren, Karin Groothuis-Oudshoorn, "**mice: Multivariate Imputation by Chained Equations in R,**" Journal of Statistical Software 45.3 (2011), pp. 1–67, 2011.

Link to paper - https://www.jstatsoft.org/article/view/v045i03

## Original Code
Github URL of the original code in R - https://github.com/stefvanbuuren/mice

Reference for python code - Iterative Imputer in https://github.com/iskandr/fancyimpute/tree/master/fancyimpute.

(This is the refactorization of the original code.)

## Description
### - Problem:
- Missing data is an important issue, even small proportions of missing data can adversely impact the quality of learning process, leading   to biased inference. This paper brings in the concept of **Multiple Imputation**. All previous standalone methods shared a common drawback, imputing a single value for one missing observation, which is then treated as the gold standard, same as the observed data in any  subsequent analysis. This implicitly assumes that imputation model is perfect and fails to account for error/uncertainty in the imputation process. This is overcome by replacing each missing value with several slightly different imputed values, refecting our uncertainty about the imputation process. This approach is called * *multiple imputation.* *

### - Architecture and Working of the Algorithm: 

![Architecture](https://github.com/karishma-chadha/ditk/blob/develop/data_cleaning/imputation/midas/readme-images/midas_arch.png)

Mice technique makes use of three functions - mice(), with() and pool().
1) mice() produces sets of imputed values(m). Here, m=3.

2) with() helps in visualizing the imputed values with help of graph plots and gives the analysed results.

3) pool() aggregates these sets of imputed values by different techniques. For example, mean, or if user wants a set of imputed values like set 1, he can pass the parameter 1 in pool().

Steps in the Algorithm:- 

Step 1: A simple imputation, such as imputing the mean, is performed for every missing value in the dataset. These mean imputations can be thought of as “place holders.”

Step 2: The “place holder” mean imputations for one variable (“var”) are set back to missing.

Step 3: The observed values from the variable “var” in Step 2 are regressed on the other variables in the imputation model, which may or may not consist of all of the variables in the dataset. In other words, “var” is the dependent variable in a regression model and all the other variables are independent variables in the regression model. These regression models operate under the same assumptions that one would make when performing (e.g.,) linear, logistic, or Poison regression models outside of the context of imputing missing data.

Step 4: The missing values for “var” are then replaced with predictions (imputations) from the regression model. When “var” is subsequently used as an independent variable in the regression models for other variables, both the observed and these imputed values will be used.

Step 5: Steps 2–4 are then repeated for each variable that has missing data. The cycling through each of the variables constitutes one iteration or “cycle.” At the end of one cycle all of the missing values have been replaced with predictions from regressions that reflect the relationships observed in the data.

Step 6: Steps 2 through 4 are repeated for a number of cycles, with the imputations being updated at each cycle. The number of cycles to be performed can be specified by the researcher.

**Mice technique is beneficial, since it performs imputation on variable-by-variable basis. Thus, different imputation models can be specified for different variables.**

### - Input and Output for Prediction and Training:
- Takes **input** as path to the gold numeric relation in csv format and introduces missingness in the dataset(20%). The algorithm is then run on this missing data.

- **Output** is the complete relation with predicted values in a newly created csv file.

- Gold dataset is used in order to evaluate the **performance of this algorithm by rmse value.**


## Evalution
- Benchmark datasets :
  
  UCI Repository Datasets
  (complete, numerical) 
 
1) Letter Recognition - http://archive.ics.uci.edu/ml/datasets/Letter+Recognition

2) Breast Cancer (Diagnostic) - http://archive.ics.uci.edu/ml/datasets/breast+cancer

3) Spam-base - http://archive.ics.uci.edu/ml/datasets/spambase


- Evaluation metrics:
  Root Mean Square Error (RMSE) 


 - Evaluation results
 
![Evaluation Results](https://github.com/karishma-chadha/ditk/blob/develop/data_cleaning/imputation/midas/readme-images/midas_eval_results.PNG) 

## Demo
- Link to the Jupyter Notebook : 

https://github.com/karishma-chadha/ditk/blob/develop/data_cleaning/imputation/mice/demo/mice_jupyter_notebook.ipynb

- Link to the video on Youtube
