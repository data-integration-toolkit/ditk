# Empower Sequence Labeling with Task-Aware Language Model

### Full Citation

Liyuan Liu, Jingbo Shang, Frank Xu, Xiang Ren, Huan Gui, Jian Peng, and Jiawei Han. 2017. Empower sequence labeling with task-aware neural language model. AAAI 2018. https://arxiv.org/pdf/1709.04109.pdf

GitHub repo: https://github.com/LiyuanLucasLiu/LM-LSTM-CRF

### Requirements
Python 3.6+
Pytorch 1.0.1+

### Instructions
To run main function for read_dataset-predict-evaluation:
```
cd <root>/ditk
pip install requirements.txt
python3 extraction/named_entity/lmlstmcrf/main.py
```

To run unit tests:
```
cd <root>/ditk
python3 extraction/named_entity/lmlstmcrf/test/test.py
```

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
The project's task is to perform named entity recognition given a sequence of words for a sentence. This model uses a novel
approach of multitasking language model sequence labeling and named entity recognition simultaneously which shared weights.
It also uses highway connections to link character level weights from the language model to the word level bi-lstm to take 
both features into account.

![FCN_schematic](figure1.png?raw=true)

### Benchmark datasets
* ConLL 2003 - https://cogcomp.org/page/resource_view/81
* Ontonotes - https://catalog.ldc.upenn.edu/LDC2013T19
* Chemdner - https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/

### Evaluation metrics and results
|         |ConLL 2003|Ontonotes|Chemdner|
|---------|----------|---------|--------|
|F1       |0.908     |0.783    |0.782   |
|Recall   |0.906     |0.752    |0.761   |
|Precision|0.907     |0.816    |0.805   |

Ontonotes and Chemdner were both trained on a subset of equal size to ConLL 2003

### Links
Notebook: https://github.com/twiet/ditk/blob/develop/extraction/named_entity/lmlstmcrf/lmlstmcrf.ipynb
