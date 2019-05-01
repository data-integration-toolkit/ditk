# Deep Residual Learning for Weakly-Supervised Relation Extraction

This repo is based on-

Yi Yao Huang and William Yang Wang, Deep Residual Learning for Weakly-Supervised Relation Extraction,2017 Association for Computational Linguistics [[paper]](https://arxiv.org/abs/1707.08866)

 
![Architecture](https://user-images.githubusercontent.com/16465582/30602043-05f63dd6-9d96-11e7-9f2e-382e15a2b37a.png)


### Original Code
https://github.com/darrenyaoyao/ResCNN_RelationExtraction

### Description
This work discuss about how we solve the noise from distant supervision. 
We propose the Deep Residual Learning for relation extraction and mitigate the influence from the noisy in semi-supervision training data.
This paper is published in EMNLP2017.

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
    
 * For predicting-
 
    Input is a sentence or a file annotated with entities and their positions.
    
    Output is the predicted relation.

### Usage
* Create object

        deepcnn=DeepResidualLearning()
       
* Read the dataset

        training_data=deepcnn.read_dataset(training_data_file)
        
* Train the model

        deepcnn.train(training_data)
        
* Evaluate the model

        deepcnn.evaluate(training_data)
        
* Predict results

        deepcnn.predict(prediction_data)
        
### Evaluation
* The model has been evaluated on three datasets, namely
    
    NYT
    
    SemEval Task8
    
    DDI 2013

## Result
![Result](https://user-images.githubusercontent.com/16465582/30602544-6c3bd1a4-9d97-11e7-9f8f-807b436ede16.png)
vector1.txt: You can use Glove vector or Word2Vec. Here is the link I used in experiment : https://drive.google.com/open?id=0B-ZjKY509crKQXA0Y2FfbFJMY0E


## Demo

* [Link to the Jupyter Notebook](./DeepResidualmain.ipynb)

* [Link to the Youtube Video](https://youtu.be/dgsUqj7Vvsg)

## Citation

      @InProceedings{huang-wang:2017:EMNLP2017,
          author    = {Huang, YiYao  and  Wang, William Yang},
          title     = {Deep Residual Learning for Weakly-Supervised Relation Extraction},
          booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
          month     = {September},
          year      = {2017},
          address   = {Copenhagen, Denmark},
          publisher = {Association for Computational Linguistics},
          pages     = {1804--1808},
          url       = {https://www.aclweb.org/anthology/D17-1191}
        }
    }
