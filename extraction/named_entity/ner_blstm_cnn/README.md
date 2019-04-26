# NER_BLSTM_CNN
- Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs
- Jason Chiu and Eric Nichols, Named entity recognition with bidirectional LSTM-CNNs, Transactions of the Association for Computational Linguistics, Volume 4, p.357-370, 2016. [original paper](https://aclweb.org/anthology/Q16-1026)


## Original Code
https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

## Description
- The `NER_BLSTM_CNN` class implements named entity reognition using bi-directional LSTM's and CNN's. The paper combines three type of features as listed below.
    - Character level features like character capitalization (allUpper, lowercase, noCaps, noInfo)
    - Additional character level features by passing through generic CNN
    - Word embeddings
- The combined features are pass through a BLSTM to generate predictions for each token

- Network Model Constructed Using Keras
 ![alt text](https://raw.githubusercontent.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/master/model.png)

## Input and Output
- Prediction
    -  Input format: Sentence (un-tokenized) - `Steve went to Paris`
    -  Output format: `[start, span, token, token, type]` - `[(None, 5, Steve, B-PER),(None, went, 4, O),(None, 2, to, O), (None, 5, Paris, B-LOC)]`
- Training
    - Input format (trainig data): `{word, POS, chunk tag, entity type}`
    - Sample data
        ```
        EU NNP B-NP B-ORG
        rejects VBZ B-VP O
        German JJ B-NP B-MISC
        call NN I-NP O
        to TO B-VP O
        boycott VB I-VP O
        British JJ B-NP B-MISC
        lamb NN I-NP O
        . . O O
        ```

## Evalution
- Benchmark Datasets
    - CoNLL-2003
    - Ontonotes 5.0
    - CHEMDNER
- Evaluation metrics
    - Precision
    - Recall
    - F-1 score
- Results (Test set)

| Dataset | Precision | Recall | F-1 score | 
| :--- | :---: | :---: | :---: | 
| CoNLL-2003-en | 0.8927 | 0.9118 | 0.9021 |  
| Ontonotes 5.0 | 0.7713 | 0.8385 | 0.8035 | 
| CHEMDNER | 0.512 | 0.497 | 0.504 | 

## Demo
- Link to the Jupyter Notebook 
- Link to the video on Youtube