## Code for EMNLP2017 paper "A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction"
Tianyu Liu, Kexiang Wang, Baobao Chang and Zhifang Sui, **A Soft-label Method for Noise-tolerant Distantly Supervised Relation
Extraction**,EMNLP 2017. ([link](https://aclweb.org/anthology/D17-1189))


## Citation

Liu, T., Wang, K., Chang, B. and Sui, Z. : A Soft-label Method for Noise-tolerant Distantly Supervised Relation. EMNLP
2017.


## Original Implementation Repository ([Repo](https://github.com/tyliupku/soft-label-RE))


## Overall Task
<p align="center"><img width="40%" src="fig.png"/></p>

### soft-label-RE
This project provides the implementation of distantly supervised relation extraction with (bag-level) soft-label adjustment.

### Distantly Supervised Relation Extraction

Distant supervision automatically generates training examples by aligning entity mentions in plain text with those in KB and labeling entity pairs with their relations in KB. If there's no relation link between certain entity pair in KB, it will be labeled as negative instance (NA).

### Multi-instance Learning
The automatic labeling by KB inevitably accompanies with wrong labels because the relations of entity pairs might be missing from KBs or mislabeled.
Multi-instances learning (MIL) is proposed to combat the noise. The method divides the training set into multiple bags of entity pairs (shown in the figure above) and labels the bags with the relations of entity pairs in the KB (**bag-level DS label**).
Each bag consists of sentences mentioning both head and tail entities.

Much effort has been made in reducing the influence of noisy sentences within the bag,
including methods based on at-least-one assumption and attention mechanisms over instances.

###  Bag-level Mislabeling
As shown in the figure above, due to the absence of (*Jan Eliasson*, *Sweden*)(*Jan Eliasson* is a Swedish diplomat.) from the *Nationality* relation in the KB,the entity pair is mislabeled as NA.

Actually, no matter how we design the weight calculation of the sentences (in that bag) for bag representation, the bag would be a noisy instance during training.
So we try to solve the problem from a different point of view. Since the bag-level DS label can be mislabeled, we design a soft-label adjustment on the bag-level DS label to correct the ill-labeled cases.

## Input/Output Format(shared within group)
* The input format is generalized for the whole Relation group
```
sentence e1 e1_type e1_start_pos e1_end_pos e2 e2_type e2_start_pos e2_end_pos relation (tab separated file)
```
* The output format is generalized for the whole Relation group
```
sentence e1 e2 predicted_relation grandtruth_relation
```
## Benchmark Dataset(shared within group)
* SemEval2010
* DDI2013
* NYT

## Evaluation metric(shared within group)
* Precision
* Recall
* F1

## Performance
|     Metric / Dataset+Method          |      Precision      | Recall | F1|
|:--------------------:|:---------------------:|:---------------------:|:---------------------:|
| NYT + ATT       |  | | |
| NYT + One       |  | | |
| SemEval2010 + One       |  | | |
| SemEval2010 + One       |  | | |
| DDI + One       |  | | |
| DDI + One       |  | | |


## Folder Structure

### Tree
```
.
├── data
|   ├── gdl.sh
|   └── getData.sh
|
├── res
|
├── tests
|   ├── test.py
|   └── testInput.txt
|
├── .gitignore
├── __init__.py
├── README.md
├── relation_extraction_3.py
├── main.py
├── model.py
├── run.ipynb
├── recipt.txt
└── requirements.txt
```

### Table
| Path                 | Description           |
|:--------------------:|:---------------------:|
| /data/gdl.sh       | Script to download from Google Cloud      |
| /data/getData.sh       | Script to download Benchmark Datasets and embedding      |
| /res       | Folder containing checkpoint, log and saved models for each run      |
| /test/*      | Unit Test Script and corresponding Test Input/Output      |
| main.py       | Invocation/Entry Point for this Relation Extraction Method   |
| model.py       | Implementation codes for this Relation Extraction Method    |
| relation_extraction_3.py       | Python3 Version Parent Class for group: Relation Extraction      |
| run.ipynb    | Jupyter Notebook file to show how to run the program |
| __init__.py      | Required file for package structure      |
| recipt.py       | Line 1: path to the code directory  Line 2:Implemented class name   |
| requirements.py       | Required python library(Not Accurate!)   |


## User Guide

### Requirements
* python 3
* tensorflow >= 1.3


### Acquire Data
Because of the size of dataset, the project provides script to download all required dataset/embedding.
Go to Data folder and run the script
```
cd data
bash getData.sh
```


### Running Program
Because this is a package method, no command line invocation provided without build & import ditk package.
In order to use it separately. You need to manually edit main.py to feed in input file.
Open main.py in root directory.
```
if __name__ == "__main__":
    # You need to add input path manually!
    # For Example: main('/home/user/Desktop/Dataset.txt')
    main()
```
You may also find the provide Youtube Video and Jupyter notebook useful.


### Youtube Video Guide
[[Youtube Video](https://youtu.be/Noj5v2ihZBA)]

### Run unit test
```
cd tests
python test.py
```
