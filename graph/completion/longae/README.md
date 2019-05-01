# Learning to Make Predictions on Graphs with Autoencoders

### Full Citation

Tran, Phi Vu. Learning to Make Predictions on Graphs with Autoencoders. Proceedings of the 5th IEEE International Conference on Data Science and Advanced Analytics (2018). https://arxiv.org/abs/1802.08352

GitHub repo: https://github.com/vuptran/graph-representation-learning

### Requirements
Python 3.6+
Tensorflow 1.13.1+

### Instructions
To run main function for read_dataset-predict-evaluation:
```
cd <root>/ditk
pip install requirements.txt
python3 graph/completion/longae/main.py
```

To run unit tests:
```
cd <root>/ditk
python3 graph/completion/longae/test/test.py
```

### Input/Output for Prediction
Input
* N x N adjacency matrix with 10% links set to \<unk\> (N is the number of nodes)
* N x F matrix of node features (F is the number of features per node)

Output
* N x N adjacency matrix with all links (N is the number of nodes)
* N x C matrix of one-hot label classes (C is the number of classes)

### Input/Output for Training
Input
* N x N adjacency matrix with 10% links set to <unk> (N is the number of nodes)
* N x F matrix of node features (F is the number of features per node)
* N x C matrix of one-hot label classes (C is the number of classes)

### Task Overview
The project is to perform multitask link prediction and node classification simultaneously using a keras autoencoder model.
Given a dataset consisting of an adjacency matrixs where one indicates reference and zero indicates lack of reference to another node,
we set a subset of links as unknown to perform training and prediction. We augment the matrix with an additional word embedding feature
for each node given a specific classification of the node as a one hot vector as well to do node classification. We combine these tasks 
together using the autoencoder model.

![FCN_schematic](figure1.png?raw=true)

### Benchmark datasets
This uses Cora and Citeseer datasets taken from https://github.com/tkipf/gcn

### Evaluation metrics and results
|   |CORA | CITESEER|
|---|-----|---------|
|AUC|0.947|0.867|
|AP |0.944|0.892|
|ACC|0.721|0.610|

### Links
Notebook: https://github.com/twiet/ditk/blob/develop/graph/completion/longae/longae.ipynb
Video: https://www.youtube.com/watch?v=L0E3LypnX7M&feature=youtu.be
