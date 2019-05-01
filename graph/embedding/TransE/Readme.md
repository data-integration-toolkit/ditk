# TransE

- Translating Embeddings for Modeling Multi-relational Data
- Bordes A., Usunier N., Garcia-Dur´an A., Weston J., and Yakhnenko O.(2013) Translating embeddings for modeling multi-relational data. In Proceedings of NIPS, 2013.

## Original Code

A general TransE implementation:

- https://github.com/thunlp/KB2E

A Python and Tensorflow implementation used in this model:

- https://github.com/ZichaoHuang/TransE

## Description

- TransE embed entities and relationships of metarelational data in low-dimensional vector spaces. TransE models relationships by interpreting them as translations operating on the low-dimensional embeddings of the entities. The task of this method includes embedding the training data, and using the embeddings to predict head/tail entities. 

- ![1556516303912](C:\Users\Sandie\AppData\Roaming\Typora\typora-user-images\1556516303912.png)

  *This image shows the idea of interpreting relation(r) as transitions from head entity(h) to tail entity(t).

- The TransE model provided in this repository contains functions to read a training file as input, to learn embeddings of the inputs, and to preform evaluations on the embeddings. 

## Input and Output

Input for embedding:

​	Files:

   - train.txt (in the format of <head_entity, tail_entity, relation>)

   - test.txt (can be split from train.txt)

   - valid.txt (can be split from train.txt)

   - entity2id.txt (<entity_name, eneity_id>)

   - relation2id.txt (<relation_name, relation_id>)

     *entity2id.txt  and relation2id.txt can be generated from train.txt by assigning each entity and relationship a unique id.

​	Parameters:

- input_folder : containing all the input files
- dimension: dimension to embed each entity and relationship to
- marginal_value: goal value to stop iteration
- batch_size: batch size for gradient descent
- max_epoch: max number of iterations



Output:

Files:

- embedded_entity.txt (in format <entity_name: embedding>)
- embedded_relation.txt(in format <relation_name: embedding>)

## Evaluation

- Benchmark datasets:

  - WN18 - WordNet dataset in format of <head_entity, tail_entity, relation>

  - FB15k - FreeBase dataset of size 15k in format of <head_entity, tail_entity, relation>

  - YAGO - original dataset in format of <head_id, relation_id, tail_id>, cleaned to format <head_entity, tail_entity, relation> for this model to use.

    |       | epoch | MeanRank | Hits@10 |
    | ----- | ----- | -------- | ------- |
    | WN18  | 1000  | 243      | 79.9    |
    | FB15k | 1000  | 546      | 36.8    |
    | YAGO  | 1000  | 5786     | 0.001   |

    

## Demo

- Jupyter Notebook: https://github.com/sandiexie-USC/ditk/blob/develop/graph/embedding/TransE/TransE.ipynb
- Youtube video: https://youtu.be/YQCG408wgLw