# UOI
- Title of the paper

    **Fast and Accurate Entity Recognition with Iterated Dilated Convolutions**
- Full citation


    Fast and Accurate Sequence Labeling with Iterated Dilated Convolution
    Emma Strubell, Patrick Verga, David Belanger, Andrew McCallum
    In Conference on Empirical Methods in Natural Language Processing (EMNLP). Copenhagen, Denmark. September 2017


## Original Code
Github URL of the original code
	[https://github.com/iesl/dilated-cnn-ner](https://github.com/iesl/dilated-cnn-ner)

## Description
    1. The paper's task is to do Named Entity Recognition on Conll2003, 
    2. The model is based on Iterated Dilated Convorlutional Neural Networks
    3. ID-CNNs permit fixed-depth convolutions to run in parallel across entire documents
    4. The model can predict each token’s label independently, or by running Viterbi inference in a chain structured graphical model.
    5. The model contains 3 layers and middle layer has dilation of 3

![Convorlutional Neural Networks](cnn.png)
*Convorlutional Neural Networks*
![Dilated Convorlutional Neural Networks](dilated)
*Dilated Convorlutional Neural Networks()

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
The output for training is model files stored under `models` folder which is the serialization for the model.
## Evalution
### Benchmark datasets
[Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/), [Ontonotes](https://catalog.ldc.upenn.edu/LDC2013T19), [CHEMDNER](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)
- Evaluation metrics and results

|Data set |Recall|Precision | F1 |
|---------|------|----------|----|
|Conll2003|  0.88|     0.89 |0.88|
|Ontonotes|  0.73|     0.73 |0.73|
|CHEMDNER|   0.68|     0.68 |0.69|
## Demo
- [Notebook](DilatedCNN.ipynb) 
- [Youtube](https://www.youtube.com/watch?v=QrIJygmQ2Ag)

## How to run the source code
### Download the dataset
In order to run the source code, the user should download a dataset as well as an embedding file.
The dataset can be download from [here](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003). 
After download the dataset, the user should put 3 text files, i.e. `test.txt`, `train.txt` and `valid.txt`
to the folder `CoNNL2003eng` and `embedding`. The folders, currently, contain sample files for guiding user to put
the current files so please overwrite the sample files.

### Python version and Tensorflow version
The user should make sure the version of Python>=3 and Tensorflow>=1.13.1

### Read the dataset
The first step is to read the dataset. The codes are as following. Please **replace** the paths by your local paths
```python
from extraction.named_entity.yiran.paper1 import Ner
d = Ner.DilatedCNN()
total_training_sample = d.read_dataset(input_files=['/home/ubuntu/dilated-cnn-ner/data/conll2003/eng.train','/home/ubuntu/dilated-cnn-ner/data/conll2003/eng.testa','/home/ubuntu/dilated-cnn-ner/data/conll2003/eng.testb'], 
                                        embedding='./data/embeddings/glove.6B.100d.txt') 
```

### Train the model
After reading the dataset, we can start to train the model. 
There will be a file added `model.h5` will be created so **please confirm python have the privilege to create a file.**
```python
d.train(data=None)

```

### Predict the files
After having model trained, we can try to predict new sentences.
```python
predict_file='dataset/CoNNL2003eng/test.txt'
result = d.predict('/home/ubuntu/dilated-cnn-ner/data/conll2003-w3-lample/eng.testb')
```
Now, we have the `tokens`, `ground_truth_tags` and `predictions`

