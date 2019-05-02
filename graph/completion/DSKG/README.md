## Title of the paper
DSKG: A Deep Sequential Model for Knowledge Graph Completion
  
## Citation
  Lingbing Guo, Qingheng Zhang, Weiyi Ge, Wei Hu1, and Yuzhong Qu. DSKG: A Deep Sequential Mode for Knowledge Graph Completion . China Conference on Knowledge Graph and Semantic Computing. [Link]:{https://arxiv.org/pdf/1810.12582v2.pdf}
  
## Input/Output Format for prediction
Entity relation Entity

## Input/Output format for training
Entity relation Entity

## Task Description
The problem being solved in the research paper is of Knowledge graph completion, it does so by entity prediction. Consider subject and a relation, the model tries to predict the best object.


## Model Description
The model takes in the input data in the form: entity relation entity. If the entity is in text format it maps entites to a number, same procedure is applied to relations, therefore, data gets converted to entity-number relation-number entity-number. Using this data, it learns the embeddings used to represent entities and relations. When test data is input, it first maps the entities and relations to numbers, then on the basis of subject entity and relation entity it outputs a number, which represents an entity. In the case of freebase entities are in form if text, therefore, we can use the entity2id file to find the corresponding entity. A description of the process is provided in the video. The test and train file in the folder can be any file of the format entity relation entity, the model handles different cases, and can learn accordingly.

## Benchmarks
Freebase and Wordnet

## Evaluation metrics and results
Hits@1 = 24.9
MRR = 33.9

## Video
https://www.youtube.com/watch?v=qnlDOFstMSY&feature=youtu.be

        
        
