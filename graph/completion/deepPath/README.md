# DeepPath: A Project for Deep Reinforcement Learning for Knowledge Graph Reasoning

## Requirements: 
- Python 3.6 (This has been upgraded from Python 2 which was the original implementation)
- Tensorflow 1.13.1
- Pre compiled transE.cpp into transX executable with command g++ transE.cpp -o transX -pthread -O3 -march=native

### Input/Output 
- input [entity, relation, entity] triples
- output list of paths

## Usage:

### read_dataset
Must be run before any other method if not using the original papers dataset. If you are using the original papers dataset skip this step and set dataset_return to be {directory_name : list of relations to be trained on}

- required_paramaters: input_file_name, the path to a knowledge graph file
- output: A dictionary with the key being the output dictionary and the value being the list of relations to be trained on.


### train
- required_parameters: A dictionary with the key being the output dictionary and the value being the list of relations to be trained on.

- optional_parameters: options={"relation": relation_name}

### predict
- required_parameters: A dictionary with the key being the output dictionary and the value being the list of relations to be tested on.

- optional_parameters: options={"relation": relation_name}

### evaluate
** Note Predict Must be run before evaluate
- required_paramters: A dictionary with the key being the output dictionary and the value being the list of relations to be tested on stored in the options array as options = {"data": dataset} where dataset is the output of read_dataset

- optional_parameters: options["relation"]= relation_name

### Benchmarks
NELL-995
FB15k-237

### Evaluation Metrics 
- MAP (FB15k-237:0.572 | NELL-995:0.796)

### Demo Video
- https://youtu.be/5M1GCCvOuuo

# Original Author
We study the problem of learning to reason in large scale knowledge graphs (KGs). More specifically, we describe a novel reinforcement learning framework for learning multi-hop relational paths: we use a policy-based agent with continuous states based on knowledge graph embeddings, which reasons in a KG vector-space by sampling the most promising relation to extend its path. In contrast to prior work, our approach includes a reward function that takes the accuravy, diversity, and efficiency into consideration. Experimentally, we show that our proposed method outperforms a path-ranking based algorithm and knowledge graph embedding methods on Freebase and Never-Ending Language Learning datasets.
Original github: https://github.com/xwhan/DeepPath/


```
@InProceedings{wenhan_emnlp2017,
  author    = {Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  title     = {DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017)},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {ACL}
}
```

## Acknowledgement
* [TransX implementations by thunlp](https://github.com/thunlp/Fast-TransX)
* [Ni Lao's PRA code](http://www.cs.cmu.edu/~nlao/)
