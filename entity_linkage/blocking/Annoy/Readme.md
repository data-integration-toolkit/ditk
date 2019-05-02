# Annoy (Approximate Nearest Neighbors Oh Yeah)
- https://github.com/spotify/annoy

## Original Code
- tmp

## Description
- Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.
- For ditk I have created a wrapper for Annoy to integrate with the ditk pipeline.
- Can specify vector ids. 
- Can search by vector id as well as by vector. 
<p align="center">
    <img src="figures/Screenshot from 2019-05-01 19-49-18.png"/>
    <p align="center">Fig. 1 Annoy Details</p>
</p>


## Functions 

### __init__
####
- Can pass all parameters needed during training/predicting/evaluating
- Returns nothing

### read_dataset
####
- Can pass path to csv file or dataset name from http://vectors.erikbern.com
- Can specify if read dataset is the train, test, or labels dataset
- Returns a pandas dataframe 

### train
####
- Takes dataframe of vectors and builds ‘num_trees’ annoy search trees
- Parameters passed in will override any predefined class member variables
- Returns annoy model

### predict
####
- Takes dataframe of vectors to predict
- Parameters passed in will override any predefined class member variables
- Returns [[Nearest Neighbor ids],[Distances]] or [[Nearest Neighbor Ids]] if include_distances = False

### evaluate
####
- Takes dataframe of ground truth neighbors and dataframe of vectors to evaluate
- Parameters passed in will override any predefined class member variables
- Returns precision, recall, and reduction ratio

## Evalution
<p align="center">
    <img src="figures/Screenshot from 2019-05-01 19-49-38.png"/>
    <p align="center">Fig. 1 Annoy Benchmark</p>
</p>


## Demo
### Jupyter Notebook
- tmp
### Video
- tmp

