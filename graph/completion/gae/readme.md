# GAE

This is an implementation of the paper "Modeling Relational Data with Graph Convolutional Networks"

## Installation

 TensorFlow (1.0 or later),
 python 2.7,
 networkx,
 scikit-learn,
 scipy

## Benchmarks

The data set that can be used has been added in the folder test. The test data could not be presented in a text file because it is an adjacency matrix. This code needs 2 adjacency matrices. For this purpose, CORA files have been added. The outher files that could be used are citeseer and pubmed.
Hence cannot use the the test sample as given common to the group.

## Run

Since this has multiple python files, main.py has been created as mentioned. 
This file is the child class of the common parent class for the group. It inherits all the methods and implements them.

To run main.py, the input data gets picked from the test folder. 
The model created gets saved temporarily in a varibale and is used to predict the test data.

## Reasons

This code has not been put in Jupiter notebook because the entire code was originally implemented in python 2 and was run on a specific version of scipy and gensim which are not compatible with jupyter notebook package. So migrating this code to Python 3 was possible, but still there was a version collision of these packages. Hence, was unable to run this on Jupyter notebook.

## Evaluation Metrics

The code has been evaluated with metrics like area under the curve(AUC) and average precision(AP)

## Citation

if you use the code, please cite:

```txt
@article{kipf2016variational,
  title={Variational Graph Auto-Encoders},
  author={Kipf, Thomas N and Welling, Max},
  journal={NIPS Workshop on Bayesian Deep Learning},
  year={2016}
}
```

## Youtube Link
https://youtu.be/Lm96cifH-wI
