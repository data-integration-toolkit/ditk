# One-Shot-Knowledge-Graph-Reasoning

An implementation of the One-Shot relational learning model described in the EMNLP 2018 paper [One-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/abs/1808.09040). The work described in this paper is focused on learning new facts, given only one training example. For example, given the example "the Arlanda Airport is located in city Stochholm", the algorithm proposed in this papers tries to automatically infer that "the Haneda Airport is located in Tokyo" by utilizing the knowledge graph information about the involved entities (i.e. the Arlanda Airport, Stochholm, the Haneda Airport and Tokyo).
### Requirements
* ``Python 3.6.5 ``
* ``PyTorch 0.4.1`` (With Cuda enabled) 
* ``tensorboardX``
* ``tqdm``

### Input/Output
* Input: A knowledge graph, containing a list of triples like [e1, r, e2]
* Output: Two lists of pairs, the first containing the found instances, the second contains the goal instances.

## Usage:

### read_dataset
Must be run before any other method if not using the original papers dataset. If you are using the original papers dataset you can set the directory for train/test data to that directory.

- required_paramaters: a list containing the input_file_name, the path to a knowledge graph file

- output: A directory containing the formatted data

### train
- required_paramaters: input_directory, this is the directory outputted by read_dataset

- output: None

### predict
- required_paramaters: input_directory, this is the directory outputted by read_dataset

- output: A file containing a dictionary containing the labeled instances of the two lists of pairs, the first containing the found instances, the second contains the goal instances.

### evaluate
- required_paramaters: input_directory, this is the directory outputted by read_dataset.
- optional_paramaters: prediction_data, this is the output of the predict step, the dictionary with the two lists of pairs.

- output: Hits@10, Hits@5, and MRR

### Benchmarks:
NELL-One, Wiki-One

### Evaluation Metrics
# NELL
- MRR: 0.329
- Hits@10: 0.405
- Hits@5: 0.367

# Wiki 
- MRR: 0.194
- Hits@10: 0.305 
- Hits@5: 0.266


### Demo Video
# https://youtu.be/gZOAzG9eX8A

### Original Author:

Reference
```
@article{xiong2018one,
  title={One-Shot Relational Learning for Knowledge Graphs},
  author={Xiong, Wenhan and Yu, Mo and Chang, Shiyu and Guo, Xiaoxiao and Wang, William Yang},
  journal={arXiv preprint arXiv:1808.09040},
  year={2018}
}
```
