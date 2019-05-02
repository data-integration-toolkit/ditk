# UOI
- Title of the paper

    **Named Entity Recognition With Parallel Recurrent Neural Networks**
- Full citation


    Named Entity Recognition With Parallel Recurrent Neural Networks
    Andrej Zukov-Gregoric, Yoram Bachrach, and Sam Coope
    the 56th Annual Meeting of the Association for Computational Linguistics (Short Papers), pages 69–74
    Melbourne, Australia, July 15 - 20, 2018. 2018 Association for Computational Linguistics


## Original Code
Github URL of the original code
	[https://github.com/PoWWoP/UOI-P18-2012](https://github.com/PoWWoP/UOI-P18-2012)

## Description
    1. The paper's task is to do Named Entity Recognition on Conll2003, 
    2. The model contains 16 parellel recurrent neural networks
    3. To reduce the total number of parameters, a single LSTM is splitted into multiple equally-sized smaller ones
    4. A final hidden concatenate layer is used to connect the parallel RNN together
    5. The paper uses GloVe for word embedding

## Dependency
- Python
- Keras
- TensorFlow
## Resources
### [GloVe](https://nlp.stanford.edu/projects/glove/)
_GloVe is an unsupervised learning algorithm for obtaining vector representations for words. 
Training is performed on aggregated global word-word co-occurrence statistics from a corpus, 
and the resulting representations showcase interesting linear substructures of the word vector space._
### [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/)
_The shared task of CoNLL-2003 concerns language-independent named entity recognition. 
We will concentrate on four types of named entities: persons, locations, 
organizations and names of miscellaneous entities that do not belong to the previous three groups._ 

## Input and Output
### Input and output for prediction
#### Input
The input file should follow the format of [Conll2012](http://conll.cemantix.org/2012/data.html). 
However, only the first 4 columns are used.

```(the word, POS tag, Chunk tag, Named entity tag)```

For example:

``Katleen NNP I-NP B-PER``
#### Output
   The output for prediction contains 3 columns
   
   ```(Token, Ground truth, Prediction)```
   
   For example:
   
   ```first B-ORDINAL B-ORDINAL```
### Input and output for Training
#### Input for training
The input for training should have the same format as prediction, where contains at least four columns ```(the word, POS tag, Chunk tag, Named entity tag)```.
#### Output for training
The output for training is ``model.h5`` which is the serialization for the model.
## Evaluation
### Benchmark datasets
[Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/), [Ontonotes](https://catalog.ldc.upenn.edu/LDC2013T19), [CHEMDNER](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)
- Evaluation metrics and results

|Data set |Recall|Precision | F1 |
|---------|------|----------|----|
|Conll2003|  0.93|     0.93 |0.93|
|Ontonotes|  0.69|     0.61 |0.65|
|CHEMDNER|   0.81|     0.82 |0.83|
## Repository Structure(On my local machine)
**This is the files on my local machine after training. Files are different from your machine** 
```
.
├── CoNNL2003eng #Embedding files
│   ├── conll2003-w3-lample
│   │   ├── test.txt
│   │   │   ├── char.txt
│   │   │   ├── examples.proto
│   │   │   ├── label.txt
│   │   │   ├── shape.txt
│   │   │   ├── sizes.txt
│   │   │   └── token.txt
│   │   ├── train.txt
│   │   │   ├── char.txt
│   │   │   ├── examples.proto
│   │   │   ├── label.txt
│   │   │   ├── shape.txt
│   │   │   ├── sizes.txt
│   │   │   └── token.txt
│   │   └── valid.txt
│   │       ├── char.txt
│   │       ├── examples.proto
│   │       ├── label.txt
│   │       ├── shape.txt
│   │       ├── sizes.txt
│   │       └── token.txt
│   ├── models  #Model files after trained
│   │   ├── checkpoint
│   │   ├── dilated-cnn
│   │   │   ├── events.out.tfevents.1556473455.tekiMacBook-Pro.local
│   │   │   ├── events.out.tfevents.1556473484.tekiMacBook-Pro.local
│   │   │   ├── events.out.tfevents.1556473672.tekiMacBook-Pro.local
│   │   │   └── graph.pbtxt
│   │   ├── dilated-cnn.tf.data-00000-of-00001
│   │   ├── dilated-cnn.tf.index
│   │   └── dilated-cnn.tf.meta
│   ├── test.txt  #CoNLL 2003 test file
│   ├── train.txt #CoNLL 2003 train file
│   ├── valid.txt #CoNLL 2003 valid file
│   └── vocabs
│       └── conll2003_cutoff_4.txt
├── Ner.py  #main implementation
├── README.md
├── Recipe.txt
├── TestCase.ipynb  #Notebook for demostration
├── embedding #GloVe
│   └── glove.6B.100d.txt
├── main.py
├── model.py
├── requirements.txt
└── tests
    ├── model.h5  #model files
    └── paper2_test.py

```
## Demo
- [Notebook](TestCase.ipynb) 
- [Youtube](https://www.youtube.com/watch?v=KJWseu_Jgnw)

## How to run the source code
### Download the dataset
In order to run the source code, the user should download a dataset as well as an embedding file.
The dataset can be download from [here](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003). 
After download the dataset, the user should put 3 text files, i.e. `test.txt`, `train.txt` and `valid.txt`
to the folder `CoNNL2003eng` and `embedding`. The folders, currently, contain sample files for guiding user to put
the current files so please overwrite the sample files.

### Python version and TensorFlow version
The user should make sure the version of Python>=3 and Tensorflow>=1.13.1

### Read the dataset
The first step is to read the dataset. The codes are as following. Please **replace** the paths by your local paths
```python
import extraction.named_entity.yiran.paper2.Ner as Ner
uoi = Ner.UOI()
train_sen, train_tag, val_sen, val_tag = uoi.read_dataset(input_files=['/home/ubuntu/UOI-P18-2012/dataset/CoNNL2003eng/train.txt',
                                                                                    '/home/ubuntu/UOI-P18-2012/dataset/CoNNL2003eng/valid.txt'],
                                                                       embedding='/home/ubuntu/UOI-P18-2012/dataset/CoNNL2003eng/glove.6B.100d.txt')
 
```

### Train the model
After reading the dataset, we can start to train the model. 
There will be a file added `model.h5` will be created so **please confirm python have the privilege to create a file.**
```python
uoi.train()

```

### Predict the files
After having model trained, we can try to predict new sentences.
```python
predict_file='dataset/CoNNL2003eng/test.txt'
tokens, ground_truth_tags, predictions = uoi.predict(predict_file)
```
Now, we have the `tokens`, `ground_truth_tags` and `predictions`

## Reference


``
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
``

``
CONLL '03 Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003 - Volume 4
``

## Authors

* **Yiran Li** - *Initial work* - [Senior Software developer](https://www.linkedin.com/in/yiran-li-0084b832/)

## License

This project is licensed under the MIT License

