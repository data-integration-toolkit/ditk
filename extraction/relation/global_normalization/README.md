
# Global Normalization
## Title
Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification.
## Citation
Heike Adel and Hinrich Schutze<br/>
Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing.<br/>
Paper:- https://aclweb.org/anthology/D17-1181

## Prediction Format
**Input** - Sentence and 2 entities.
**Output** - Relation between the entities.

## Training Format
**Input:**  Two input files - *.txt* files for training and testing.

**Output:** 
* Model performs global normalization on the CRF layer and selects the one that gives the best F1 score for named entity and relation extraction.
* It uses the result of best epoch to predict relations for test data and store it in the output directory.


## Approach
### Overview
Named entity types and relations are often mutually dependent. If the types of entities are known, the search space of possible relations between them can be reduced and vice versa. This model uses a globally normalized convolutional neural networks for joint entity classification and relation extraction.

### Steps
1. Modeling context and entities:-
      * Input tokens:- Word embeddings trained on Wikipedia with word2vec. 
      * Identifying the class of an entity ek, the model uses the context to its left, the words constituting ek and the context to its right
	  * Classifying the relation between two entities ei and ej , the sentence is split into six parts: left of ei, ei, right of ei, left of ej , ej , right of ej .

2. Sentence Representation
      * One CNN layer for convolving the entities and one for the contexts.
      * k-max pooling for both the entities and the contexts and concatenate the results.

3. Global Normalization Layer
	* Linear chain CRF layer- Joint entity and relation classification problem as a sequence of length three for the CRF layer.

### Experimental Setup
The experiment is carried out in two different setups:-
* Setup 1:- Separate models for EC and RE on the ERR dataset. For RE, they only identify relations between named entity pairs. 
* Setup 2:- Table Filling Task. Cell (i, j) contains the relation between word i and word j.

## Steps to run the Model
**main.py:** Entire model is triggered.
*  Provide Train and Test datafile in .txt format.
*  Returns the predicted relations for Test file in output directory.
	
**Global_Normalization.py:** Provide implementaion for text_similarity_2.py superclass.
* read_dataset() - Read the dataset from the file location as well as the config parameters.
* train()
	* Train the model and selects the best epoch (Best average F1 score for named entity and relation extraction).
	* Predicts the relation for Test file from the best epoch selected above and store the results in output directory.
	* Calculates the F1 score from the best epoch.
* No evaluate() and predict() module.
	
**configs/config.crf.setup1**
**configs/config.crf.setup2**
* Parameters changes as per the database name. Currently configured to handle CoNLL04, SemEval2010 and NYT dataset.
	
## Benchmark Datasets
* CoNLL04
* SemEval 2010
* NYT

## Evaluation Metrics
* Precision
* Recall
* F1 score

## Results obtained
### Setup 1
| Dataset  | Average F1 score for NE and RE|
| ------------- | ------------- |
| CoNLL04  | 0.76738  |
| SemEval 2010  | 0.63452  |

### Setup 2
| Dataset  | Average F1 score for NE and RE|
| ------------- | ------------- |
| CoNLL04  | 0.73598  |
| SemEval 2010  | 0.58452  |

## Jupyter notebook

https://colab.research.google.com/drive/1EFN0dqnsYezu9DmnwIJ7pWDA6ZrSGmeD

## YouTube links
https://youtu.be/qop8a5yld2Q
