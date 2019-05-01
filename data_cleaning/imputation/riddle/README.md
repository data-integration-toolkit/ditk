# Method Name
- RIDDLE: Race and ethnicity Imputation from Disease history with Deep LEarning
- Ji-Sung Kim, Xin Gao, Andrey Rzhetsky, Published in PLOS Computational Biology 2018 <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006106>

## Original Code
<https://github.com/jisungk/riddle>
<https://riddle.ai/>

## Description
- RIDDLE (Race and ethnicity Imputation from Disease history with Deep LEarning) is an open-source Python2 (converted to Python 3) library for using deep learning to impute race and ethnicity information in anonymized electronic medical records (EMRs). RIDDLE provides the ability to (1) build models for estimating race and ethnicity from clinical features, and (2) interpret trained models to describe how specific features contribute to predictions. RIDDLE uses Keras to specify and train the underlying deep neural networks. The default architecture is a deep multi-layer perceptron (deep MLP) that takes binary-encoded features and targets.

## Input and Output
- Prediction Input - table of values. Prediction Output - table of values 
- Training Input - table of values. Output - table of values 

## Evalution
- BREAST CANCER <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer> 
- Mean Squared Error: 0.4035

- SPAM <https://archive.ics.uci.edu/ml/datasets/spambase>
- Mean Squared Error: 0.3880


- LETTER RECOGNITION <https://archive.ics.uci.edu/ml/datasets/letter+recognition>
- Mean Squared Error: 0.3880

## Demo
- Link to the Jupyter Notebook 
- Link to the video on Youtube
