# Python 3 code for GAIN (Generative Adversarial Imputation Network)

## GAIN: Missing Data Imputation using Generative Adversarial Nets, ICML 2018
Authors: Jinsung Yoon, James Jordon, Mihaela van der Schaar

This is a refactored implementation of https://github.com/jsyoon0823/GAIN

A novel method for imputing missing data by adapting the well-known Generative Adversarial Nets (GAN) framework

Paper Link:  http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf

## Input/Output format for training and prediction

Input: .csv file without headers, numerical values

Output: .csv file generated in gain/ directory 

## Approach – ADVERSARIAL training / GAN 

The generator (G) observes some components of a real data vector, imputes the missing components conditioned on what is actually observed, and outputs a completed vector. 

The discriminator (D) then takes a completed vector and attempts to determine which components were actually observed and which were imputed. 

The discriminator is trained to minimize the classification loss (when classifying which components were observed and which have been imputed) 

The generator is trained to maximize the discriminator’s misclassification rate. Thus, these two networks are trained using an adversarial process. 

## Running the code

Please refer to gain/demo/GAIN.ipynb to help you start the process 
    
	Create a GAIN object with required parameters 
	    gain_obj = GAIN(128, 0.2, 0.9, 10, 0.8)
	
	Make sure data is present in .csv form in data folder or the path you are specifiying 
	
	Invoke gain_obj.train() to train the GAN 
	
	Invoke gain_obj.test() to impute and evaluate

## Data 

Make sure data is in the folder, for this code, we've used Letter.csv, Spam.csv 

## Source Code

The source code is the gain/src/
GAIN.py file

## Tests

Unit tests have been provided in the tests/ folder

## Benchmarking Datasets

The following UCI repository datasets have been used: 

	Breast Cancer Wiscosnin 
	Spambase
	Letter

## Evaluation Results (RMSE) 

Dataset | GAIN  | DITK/GAIN |
|-----------| ------------- | ------------- |
| UCI Breast Cancer Dataset | 0.0546 | 0.212 |
| UCI Spam Dataset | 0.0513 | 0.0573 | 
| UCI Letters Dataset | 0.1198 | 0.1249 | 

## Helpful demo links

Jupyter Notebook: https://github.com/aru-jo/ditk/blob/develop/data_cleaning/imputation/gain/demo/GAIN.ipynb

Youtube Link: Yet to be posted

