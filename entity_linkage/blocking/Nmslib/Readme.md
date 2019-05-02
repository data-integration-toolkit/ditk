# Non-Metric Space Library (NMSLIB)
- NMSLIB Citation: 
@inproceedings{DBLP:conf/sisap/BoytsovN13,
  author    = {Leonid Boytsov and
               Bilegsaikhan Naidan},
  title     = {Engineering Efficient and Effective Non-metric Space Library},
  booktitle = {Similarity Search and Applications - 6th International Conference,
               {SISAP} 2013, {A} Coru{\~{n}}a, Spain, October 2-4, 2013, Proceedings},
  pages     = {280--293},
  year      = {2013},
  crossref  = {DBLP:conf/sisap/2013},
  url       = {https://doi.org/10.1007/978-3-642-41062-8\_28},
  doi       = {10.1007/978-3-642-41062-8\_28},
  timestamp = {Thu, 25 May 2017 00:42:36 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/sisap/BoytsovN13},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
- HNSW Citation: 
@article{DBLP:journals/corr/MalkovY16,
  author    = {Yury A. Malkov and
               D. A. Yashunin},
  title     = {Efficient and robust approximate nearest neighbor search using Hierarchical
               Navigable Small World graphs},
  journal   = {CoRR},
  volume    = {abs/1603.09320},
  year      = {2016},
  url       = {http://arxiv.org/abs/1603.09320},
  archivePrefix = {arXiv},
  eprint    = {1603.09320},
  timestamp = {Mon, 13 Aug 2018 16:46:53 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/MalkovY16},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

## Original Code
- https://github.com/nmslib/nmslib

## Description
- Non-Metric Space Library (NMSLIB) is an efficient cross-platform similarity search library and a toolkit for evaluation of similarity search methods. The core-library does not have any third-party dependencies.
- For ditk I have created a wrapper for NMSLIB to easily use their state of the art Hierarchical Navigable Small World graph (HNSW) implementation.
<p align="center">
    <img src="figures/Screenshot from 2019-05-01 19-48-33.png"/>
    <p align="center">Fig. 1 NMSLIB Details</p>
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
- Takes dataframe of vectors and builds a Hierarchical Navigable Small World
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

## Parameter Descriptions
### M
- M: the number of bi-directional links created for every new element during construction. Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic dimensionality and/or high recall, while low M work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter also determines the algorithm's memory consumption, which is roughly M * 8-10 bytes per stored element.

### efc
- efc: the parameter has the same meaning as ef, but controls the index_time/index_accuracy. Bigger efc leads to longer construction, but better index quality. At some point, increasing efc does not improve the quality of the index. One way to check if the selection of efc was ok is to measure a recall for M nearest neighbor search when ef =efc: if the recall is lower than 0.9, than there is room for improvement.



## Evalution
<p align="center">
    <img src="figures/Screenshot from 2019-05-01 19-48-54.png"/>
    <p align="center">Fig. 1 Nmslib Benchmark</p>
</p>
Here we see that Nmslib’s hnsw implementation performs best in both recall and inference frequency.


## Demo
### Jupyter Notebook
- See src/NmslibDemo.ipynb
### Video
- https://www.youtube.com/watch?v=ycZ5xbK1BBg&feature=youtu.be

