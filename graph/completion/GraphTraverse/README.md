## Title of the paper
Traversing Knowledge Graphs in Vector Space
  
## Citation
Kelvin Gu, John Miller, Percy Liang. Traversing Knowledge Graphs in Vector Space. arXiv preprint arXiv:1506.01094. 
https://arxiv.org/pdf/1506.01094v2.pdf

  
## Input/Output Format for prediction
Entity relation Entity

## Input/Output format for training
Entity relation Entity

## Task Description
The problem being solved in the research paper is of Knowledge graph completion, it does so by entity prediction. Consider subject and a relation, the model tries to predict if the object present in the test data, subject relation object, could be valid or not.


## Model Description
The Model forms a matrix of glove vectors of entity relation object and on the basis of these vectors it learns a score function according to the optimization technique discussed in the paper. In the test data, it first forms the glove vectors of the entity relation object, and then on the basis of these vectors it tries to predict if the given test data can be valid or not, it does that using its score function. If the score function is greater than 0 it's valid, else it's not. In this way it does knowledge graph completion, in the form of question answering technique.


## Benchmarks
Freebase and Wordnet

## Evaluation metrics and results
Mean Quantile = 92.8
Hits@10= 78.6

## Video
https://www.youtube.com/watch?v=dzSeD3w3NbA&feature=youtu.be

        
        
