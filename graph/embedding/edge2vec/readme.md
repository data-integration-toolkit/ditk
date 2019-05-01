# Edge2vec

This is an implementation of the paper "edge2vec: Representation learning using edge semantics for biomedical knowledge discovery". In this paper, we propose the edge2vec model, which represents graphs considering edge semantics. An edge-type transition matrix is trained by an Expectation-Maximization approach, and a stochastic gradient descent model is employed to learn node embedding on a heterogeneous graph via the trained transition matrix.

## Installation

install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'argparse',
        'sklearn',
        'math',
        'gensim',
        'networkx'
    ]

## Benchmarks

The data set that can be used is the test sample data given for the graph embedding group.

```txt
entity_id relation_id entity_id
```

## Run

Since this has multiple python files, main.py has been created as mentioned. 
This file is the child class of the common parent class for the group. It inherits all the methods and implements them.

To run main.py, input.txt has to be placed in the same directory as main.py. 
The transition model gets saved in the file matrix.txt which is an edge type matrix. This model is hence used to get the final node ebmedding which is saved in a file vector.txt.

## Reasons

This code has not been put in Jupiter notebook because the entire code was originally implemented in python 2 and was run on a specific version of scipy and gensim which are not compatible with jupyter notebook package. So migrating this code to Python 3 was possible, but still there was a version collision of these packages. Hence, was unable to run this on Jupyter notebook.

## Evaluation Metrics

This paper implemetation originally did not have any evaluation metric. I came up with cosine similarity between the entitites in the input. This metric gives a visualisation on how closely similar the entities are depending on if they have an edge between them.

## Citation

if you use the code, please cite:

Gao, Zheng, Gang Fu, Chunping Ouyang, Satoshi Tsutsui, Xiaozhong Liu, and Ying Ding. 
"edge2vec: Learning Node Representation Using Edge Semantics." arXiv preprint arXiv:1809.02269 (2018).

## Youtube Link
https://youtu.be/dUhOWAifjMM
