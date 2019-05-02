# Missing Data Imputation for Supervised Learning

* **Title of the paper:** : Missing Data Imputation for Supervised Learning.
* **Full Citation:** Jason Poulos, Raefel Valle. International Journal on Applied Artificial Intelligence, Vol 32, Berkeley, CA, 2018.

## Original Code
https://github.com/rafaelvalle/MDI

## Description
Missing data imputation can help improve the performance of prediction models in situations where missing data hide useful information. Comparison of methods for imputing missing categorical data for supervised classification tasks are explained here. We experiment on benchmark datasets with missing data, comparing classifiers trained on non-imputed (i.e., one-hot encoded) or imputed data with different levels of additional missing-data perturbation. We show imputation methods can increase predictive accuracy in the presence of missing-data perturbation, which can actually improve prediction accuracy by regularizing the classifier. 

**Imputation Methods**

1. Random Replace
2. Summary (Mode)
3. SVM
4. Random Forest
5. Logistic Regression

**Classifiers**

1. Random Forest
2. Decision Trees

## Approach

**1. Instantiate Imputation_with_Supervised_Learning class**
```bash
obj = Imputation_with_supervised_learning()
```
**2. Read dataset**
If header is present in input file, header flag is set to True. If the file has missing values, the missing_values flag should be set to True. If the dataset has categorical values, categorical_values flag should be set to True.
```bash
dataname = 'breast_cancer'
input_data = obj.preprocess(filename, header, missing_values, categorical_values)
```

**3. Prepare Training data**<br/>
The dataset is split into 2/3 for train and 1/3 for test. The dataset has different levels of additional missing-data perturbation and imputation is done for each of levels of additional missing-data perturbation. It also stores scaler objects to be used on the test set.
```bash
train_data, labels_train, test_data, labels_test = obj.train(input_data, dataname)
```
**4. Prepare Testing data**<br/>
1/3 of the dataset is used for testing. The test dataset is scaled and binarized for all categorical columns.
```bash
obj.test(test_data, labels_test, dataname)
```
**5. Perform imputation on dataset read in preprocess function**<br/>
The impute command imputes missing values in the dataset.
```bash
obj.impute(input_data)
```
**6. Evaluation**<br/>
Loads the complete input dataset, imputed table and calculates the performance on the input using RMSE(Root Mean Squared Error).
```bash
obj.evaluate(filename)
```
**7. Predict labels in case of classification task**<br/>
The predict command predicts the labels in the test dataset. Classifiers used are Random Forest and Decision Tree. Different hyperparameters are used. The evaluation metric used is test set error rate in this case which is printed as and when the MCAR perturbation ratio and hyperparameters change.
```bash
obj.predict(dataname)
```


## Input and Output
* Input: Table with missing data (.csv)
* Output: Table with imputed data (.csv)

## Benchmark Datasets and Evaluation
* **UCI Spam Dataset**
  * 4601 rows * 52 columns
  * contains features about emails
  * class label:  spam/not spam

* **UCI Breast Cancer Dataset**
  * 569 rows * 32 columns
  * contains features from digitized images
  * class label: benign/malignant
  
* **UCI Letters Dataset**
  * 20000 rows * 16 columns
  * features are various parameters for 26 letters of English alphabet


**Evaluation Metric** : Root Mean Square Error(RMSE)

| Dataset                   | MICE   | MissForest | Matrix | VAE    | EM     | DITK   |
|---------------------------|--------|------------|--------|--------|--------|--------|
| UCI Spam Dataset          | 0.0699 | 0.0553     | 0.0542 | 0.0670 | 0.0712 | 0.0839 |
| UCI Letters Dataset       | 0.1537 | 0.1605     | 0.1442 | 0.1351 | 0.1563 | 0.1485 |
| UCI Breast Cancer Dataset | 0.0646 | 0.0608     | 0.0946 | 0.0697 | 0.0634 | 0.1069 |

Evaluation Results: RMSE average calculated on evaluation dataset at 20% random missing rate


## Demo

* Jupyter Notebook: https://github.com/samikshm/ditk/blob/develop-py2/data_cleaning/imputation/imputation_supervised_learning/imputation_supervised.ipynb
* Youtube video: https://youtu.be/ItH8MOj0R1k
