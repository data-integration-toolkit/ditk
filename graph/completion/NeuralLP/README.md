# Neural Logic Programming

This is the implementation of Neural Logic Programming for Knowledge Graph Completion, proposed in the following paper:

[Differentiable Learning of Logical Rules for Knowledge Base Reasoning](https://arxiv.org/abs/1702.08367).
Fan Yang, Zhilin Yang, William W. Cohen.
NIPS 2017.

## Requirements
- Python 2.7
- Numpy 
- Tensorflow 1.0.1

## Original Code
https://github.com/fanyangxyz/Neural-LP

## Input and Output

Input:

This model takes as input a set of entity relation entity triples
The datasets used are WordNet 18 and FreeBase 15K
For Quick start purposes, two datasets family and kinship have been included, to show the working of the model

Output:

The model outputs a set of predicted entity relation entity triples (the complete knowledge graph)

## Description

Task:
Use a Differentiable Model to learn logical rules for Knowledge Base Completion

Approach:
1. This paper aims at learning probabilistic first order logic rules for Knowledge Base Reasoning
2. It uses Neural Logic Programming to learn these logic rules in an end to end differentiable model
3. These learned rules are then used for Knowledge Base Completion


## Evaluation

The benchmark datasets used are WordNet 18 and Freebase 15K
The datasets contain rows of the form (entity. relation, entity)

The evaluation metrics used are Hits@10, Mean Rank and Mean Reciprocal Rank

Results:

WN18:

1. Hits@10: 98.1
2. MRR: 0.76

FB15K:

1. Hits@10: 65.3
2. MRR: 0.56

## To Run

Execute the main.py file:

1. To the read_dataset method, pass the path to where the datasets are saved
    (ensure the folder being passed contains facts, train, test, and valid files)
2. Pass the same path (where the datasets are saved) to the get_truths method as well

## Quick start
The following command starts training a dataset about family relations, and stores the experiment results in the folder `exps/demo/`.

```
python src/main.py --datadir=datasets/family --exps_dir=exps/ --exp_name=demo
```

Wait for around 8 minutes, navigate to `exps/demo/`, there is `rules.txt` that contains learned logical rules. 
