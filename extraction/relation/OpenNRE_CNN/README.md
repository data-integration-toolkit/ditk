Student Name: Debankur Ghosh
# Neural Relation Extraction with Selective Attention over Instances, Annual meeting of the Association for Computational Linguistics 


### Requirements<br>
* Python (>=2.7)
* Numpy (>=1.13.3)
* TensorFlow (>=1.4.1)
* CUDA (>=8.0) if you are using gpu
* Matplotlib (>=2.0.0)
* scikit-learn (>=0.18)

### Input Data
* SemEval2010
* DDI2013
* NYT

### Data Format

For training and testing, you should provide four JSON files including training data, testing data, word embedding data and relation-ID mapping data.

* Training Data & Testing Data

Training data file and testing data file, containing sentences and their corresponding entity pairs and relations, should be in the following format

[
    {
        'sentence': 'Bill Gates is the founder of Microsoft .',
        'head': {'word': 'Bill Gates', 'id': 'm.03_3d', ...(other information)},
        'tail': {'word': 'Microsoft', 'id': 'm.07dfk', ...(other information)},
        'relation': 'founder'
    },
    ...
]
IMPORTANT: In the sentence part, words and punctuations should be separated by blank spaces.

* Word Embedding Data

Word embedding data is used to initialize word embedding in the networks, and should be in the following format

[
    {'word': 'the', 'vec': [0.418, 0.24968, ...]},
    {'word': ',', 'vec': [0.013441, 0.23682, ...]},
    ...
]
* Relation-ID Mapping Data

This file indicates corresponding IDs for relations to make sure during each training and testing period, the same ID means the same relation. Its format is as follows

{
    'NA': 0,
    'relation_1': 1,
    'relation_2': 2,
    ...
}
IMPORTANT: Make sure the ID of NA is always 0.

### Input format for training/prediction
* The input format is generalized for the whole Relation group
```
sentence e1 e1_type e1_start_pos e1_end_pos e2 e2_type e2_start_pos e2_end_pos relation (separated by tab)
```
* The output format is generalized for the whole Relation group
```
sentence e1 e2 predicted_relation grandtruth_relation
```

### Data Preparation 
* Please place data in /data folder with folder name and change the cnn.py with the data name 

### Download pretrain model from [here](https://drive.google.com/file/d/1eSGYObt-SRLccvYCsWaHx1ldurp9eDN_/view?usp=sharing) <br>



### cnn.py
* Create model
```

```
* Load dataset
```
with open("data/nyt/rel2id.json") as f:
    rel2id = json.load(f)

with open("test_result/nyt_pcnn_ave_pred.json") as f:
    test_result = json.load(f)
```


### Benchmark datasets
* SemEval2010
* Precision = 0.86
* Recall = 0.89
* F1 = 87.49
<br>
* DDI2013
* Precision = 0.67
* Recall = 0.80
* F1 = 72.85
<br>
* NYT
* Precision = 0.80
* Recall = 0.20
* F1 = 32.18

### jupyter file
* playground.ipynb

### Demo video
*  [video](https://www.youtube.com/watch?v=_qNI1YaPIh0&feature=youtu.be)<br>

### Citation
```
Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Neural Relation Extraction with Selective Attention over Instances, Annual meeting of the Association for Computational Linguistics (ACL), Berlin, Germany, 2016 
GitHub: https: //github.com/thunlp/NRE

```
