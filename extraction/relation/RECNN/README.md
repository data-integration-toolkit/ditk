# Convolutional Neural Networks for Relation Extraction

This repo is based on-

Thien Huu Nguyen and Ralph Grishman, Relation Extraction: Perspective from Convolutional Neural Networks. Proceedings of NAACL-HLT,  Denver, Colorado, 2015.
 [[paper]](http://www.aclweb.org/anthology/W15-1506)
<p align="center">
	<img width="700" height="400" src="https://user-images.githubusercontent.com/15166794/32838125-475cbdba-ca53-11e7-929c-2e27f1aca180.png">
</p>

### Original Code
https://github.com/roomylee/cnn-relation-extraction

### Description
We present a CNN for relation extraction that emphasizes an unbalanced corpus and minimizes usage of external supervised NLP toolkits for features.
The network uses multiple window sizes for filters,
position embeddings for encoding relative distances
and pre-trained word embeddings for initialization
in a non-static architecture. 

### Requirements
* Python 3
* Tensorflow >= 1.6
* Numpy

### Input and Output
* For Training-

    Input is a sentence or a file annotated with entities and their positions.
    
        sentence e1 e1_type e1_start_pos e1_end_pos e2 e2_type e2_start_pos e2_end_pos relation (separated by tab)
    Output is the predicted relation.
    
        sentence e1 e2 predicted_relation ground_relation
    
 * For Prediction-
 
    Input is a sentence or a file annotated with entities and their positions.
    
    Output is the predicted relation.

### Usage
* Create object

        relationextraction=RelationExtractionCNN()
       
* Read the dataset

        training_data=relationextraction.read_dataset(training_data_file)
        
* Train the model

        relationextraction.train(training_data)
        
* Evaluate the model

        relationextraction.evaluate(training_data)
        
* Predict results

        relationextraction.predict(training_data)
        
### Evaluation
* The model has been evaluated on three datasets, namely
    
    NYT
    
    SemEval Task8
    
    DDI 2013

## Results
#### Official Performance
![performance](https://user-images.githubusercontent.com/15166794/47507952-24510a00-d8ae-11e8-93e1-339e19d0ab9c.png)

#### Learning Curve (Accuracy)
![acc](https://user-images.githubusercontent.com/15166794/47508193-988bad80-d8ae-11e8-800c-4f369cf23d35.png)

#### Learning Curve (Loss)
![loss](https://user-images.githubusercontent.com/15166794/47508195-988bad80-d8ae-11e8-82d6-995367bc8f42.png)


## Demo

* [Link to the Jupyter Notebook](./RECNNmain.ipynb)

* [Link to the Youtube Video](https://youtu.be/DMSYXp6TPSI)
