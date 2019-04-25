# ANALOGY
### Graph Embedding Project for CSCI548@USC
This code base implements the following paper:
##### Hanxiao Liu, Yuexin Wu, Yiming Yang. Analogical Inference for Multi-relational Embeddings. Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70, 2017. Paper: https://arxiv.org/pdf/1705.02426.pdf
##### This code base simplifies/refactors this GitHub repo: https://github.com/mana-ysh/knowledge-graph-embeddings 
#### Datasets:
This code has been tested on the FB15k and WN18 datasets. 


Running the code:

1. Initialize a model
```python
algorithm = ANALOGY()
```
2. Load the Data Set
```python
    train_file_names = {"train": input_file_path + "train.txt",
# Optional              "valid": input_file_path + "valid.txt",
# Optional              "whole": input_file_path + "whole.txt",
                        "relations": input_file_path + "relation2id.txt",
                        "entities": input_file_path + "entity2id.txt"}

    algorithm.read_dataset(train_file_names)
```
3. Learn the embeddings (i.e train model)
```python
parameters = {"mode": 'single',
              "epoch": 3,
              "batch": 128,
              "lr": 0.05,
              "dim": 200,            
              "negative": 3,         
              "opt": 'adagrad',
              "l2_reg": 0.001,
              "gradclip": 5,
              'filtered': True}

algorithm.learn_embeddings(parameters)
```
