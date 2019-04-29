# Empower Sequence Labeling with Task-Aware Language Model

### Full Citation

Liyuan Liu, Jingbo Shang, Frank Xu, Xiang Ren, Huan Gui, Jian Peng, and Jiawei Han. 2017. Empower sequence labeling with task-aware neural language model. AAAI 2018. https://arxiv.org/pdf/1709.04109.pdf

GitHub repo: https://github.com/LiyuanLucasLiu/LM-LSTM-CRF

### Input/Output for Prediction
Input
Space separated format generalized for the group.
Index 0 - Text
Index 3 - Named Entity Label
```
EU NNP I-NP I-ORG - - - - - - - -
rejects VBZ I-VP O - - - - - - - -
German JJ I-NP I-MISC - - - - - - - -
call NN I-NP O - - - - - - - -
to TO I-VP O - - - - - - - -
```

Output
```
text true_label prediction 
```

### Input/Output for Training
Same as above

### Task Overview
The project is to perform multitask link prediction and node classification simultaneously using a keras autoencoder model.
Given a dataset consisting of an adjacency matrixs where one indicates reference and zero indicates lack of reference to another node,
we set a subset of links as unknown to perform training and prediction. We augment the matrix with an additional word embedding feature
for each node given a specific classification of the node as a one hot vector as well to do node classification. We combine these tasks 
together using the autoencoder model.

![FCN_schematic](figure1.png?raw=true)

### Benchmark datasets
* ConLL 2003 - https://cogcomp.org/page/resource_view/81
* Ontonotes - https://catalog.ldc.upenn.edu/LDC2013T19
* Chemdner - https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/

### Evaluation metrics and results
|         |ConLL 2003|Ontonotes|Chemdner|
|---------|----------|---------|--------|
|F1       |0.914     |0.783    |0.782   |
|Recall   |0.916     |0.752    |0.761   |
|Precision|0.913     |0.816    |0.805   |

Ontonotes and Chemdner were both trained on a subset of equal size to ConLL 2003

### Links
Notebook: https://github.com/twiet/ditk/blob/develop/graph/completion/longae/longae.ipynb
