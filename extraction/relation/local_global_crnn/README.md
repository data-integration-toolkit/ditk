## Code for CoNLL2017 paper "Learning local and global contexts using a convolutional recurrent network model for relation classification in biomedical text"
Desh Raj and Sunil Kumar Sahu and Ashish Anand, **Learning local and global contexts using a convolutional recurrent
network model for relation classification in biomedical text**,CoNLL 2017. ([link](https://www.aclweb.org/anthology/K17-1032))


## Citation

Raj, D., Sahu, S. and Anand, A: Learning local and global contexts using a convolutional recurrent network model for
relation classification in biomedical text. CoNLL 2017.


## Original Implementation Repository ([Repo1](https://github.com/kwonmha/Convolutional-Recurrent-Neural-Networks-for-Relation-Extraction)) ([Repo2](https://github.com/desh2608/crnn-relation-classification))


## Overall Task
<p align="center">
	<img width="700" height="400" src="https://user-images.githubusercontent.com/8953934/39967385-05995058-56f5-11e8-8080-73d8098cab6b.JPG">
</p>

### Embedding
Pretrained word vectors are used as inputs for most of such model. These embeddings capture the semantic similarity between words in a global context better than one-hot representations.
In this implementaion, we use **/GoogleNews-vectors-negative300.bin**, please refer to guidance on how to download in Data Acquisition Section.

### RCNN
* Stands for combination of Recurrent Neural Network and Convolution Neural Network
* RNNs utilize the word order in the sentence, and are also able to learn the long-term dependencies
* CNNs are capable of learning local features such as short phrases or recurring n-grams, similar to the way they provide translational, rotational and scale invariance in vision

### Proposed Method
The architecture of this model is:
*  Embedding layer
*  Recurrent layer
*  First pooling layer
*  Convolution layer
*  Second pooling layer
*  Max pooling over time / Attention-based pooling
*  Fully connected and softmax


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
| NYT + Max       |69.56  |66.61 |68.05 |
| NYT + Att      | 66.84 | 62.48|68.05 |
| SemEval2010 + Max      |73.14 | 72.68 |72.91 | 
| SemEval2010 + Att        |66.95  | 68.99| 70.14|
| DDI + Max       | 71.94 |60.17 |64.95 |
| DDI + Att       | 67.57 |58.87 |62.35 |


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
Go to Data folder and run the script(Note that Downloading Google News Embedding could take a long time )
```
cd Data
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
You may also find the provided Youtube Video and Jupyter notebook useful.


### Youtube Video Guide
[[Youtube Video](https://youtu.be/JtQrqSt24uk)]

### Run unit test
```
cd tests
python test.py
```
