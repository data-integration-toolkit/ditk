# Edge2vec

This is an implementation of the paper "edge2vec: Representation learning using edge semantics for biomedical knowledge discovery"

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

## Dataset

The data set that can be used is the test sample data given for the graph embedding group.

```txt
entity_id relation_id entity_id
```

## Run

Since this has multiple python files, main.py has been created as mentioned. 
This file is the child class of the common parent class for the group. It inherits all the methods and implements them.

To run main.py, input.txt has to be placed in the same directory as main.py. 
The transition model gets saved in the file matrix.txt and the final embedding gets saved in a file vector.txt.

## Reasons

This code has not been put in Jupiter notebook because the entire code was originally implemented in python 2 and was run on a specific version of scipy and gensim which are not compatible with jupyter notebook package. So migrating this code to Python 3 was possible, but still there was a version collision of these packages. Hence, was unable to run this on Jupyter notebook.

## Citation

Citations
if you use the code, please cite:

Gao, Zheng, Gang Fu, Chunping Ouyang, Satoshi Tsutsui, Xiaozhong Liu, and Ying Ding. 
"edge2vec: Learning Node Representation Using Edge Semantics." arXiv preprint arXiv:1809.02269 (2018).

## License
The code is released under BSD 3-Clause License
