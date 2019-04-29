# NER with BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Devlin, Jacob et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**, CoRR abs/1810.04805, 2018<br>
This repository contains the source code for the NER system presented in the following research publication ([link](https://arxiv.org/pdf/1810.04805.pdf))

### Requirements<br>
* python 3.6
* tensorflow >= 1.6

### Input Data
* CoNLL2003
* OntoNotes2012

### Input format for training/prediction
* The input format is generalized for the whole NER group
* It contains multiple columns got from CoNLL2003 and CoNLL 2012 (separated by space)
* Sample
* ![Common Input](CommonInput.png)
```
Yes UH (TOP(S(INTJ*) O bc/cnn/00/cnn_0003 0 0 - - - Linda_Hamilton * -
they PRP (NP*) O bc/cnn/00/cnn_0003 0 1 - - - Linda_Hamilton * (15)
did VBD (VP*) O bc/cnn/00/cnn_0003 0 2 do 01 - Linda_Hamilton (V*) -
```
* Common output
```
word entity predicted_entity
```

### Data Preparation 
* CoNLL2003 - Find data from ./conll2003_NERData and Run conll2003_to_ditk function in convert_to_2003.py
```
main():
  lines = [line.strip() for line in open('./conll2003_NERdata/train.txt')]
  ret = conll2003_to_ditk(lines)
  with open('./NERdata/dev_test.txt', 'w') as f:
    for line in ret:
      f.writelines(' ".join(line)+'\n')
```
* OntoNotes2012 - Download data from [here](https://drive.google.com/open?id=1OauoEoPONWgwV3vH759uoBdP7MQRkr9N) and unzip the file in NERdata directory.<br>

### Download pretrain model
* Download pretrain bert model from [here](https://drive.google.com/open?id=1UBgb9OlLFvYGzpUufaj9Voe36muxW4Ga) and unzip the file in bert_ner directory.
* Download pretrain model from [here](https://drive.google.com/open?id=1ZNj9uXPKv1jWtla0ur2JQg2Y5S-g9LgL) and unzip the file in bert_ner directory.



### main.py
* Create model
```
my_model = BERT_Ner()
```
* Load dataset
```
file_dict = {
    "train": {
        "data": "./NERdata/train.txt"
    },
    "dev": {
        "data": "./NERdata/dev.txt"
    },
    "test": {
        "data": "./NERdata/test.txt"
    }
}
train_data, test_data, dev_data = my_model.read_dataset(file_dict, 'ditk')
```
* Train model - Trained model automatically saved at ./output/result_dir
```
my_model.train(train_data)
```
* Predict
```
res = my_model.predict(test_data)
```
* Evaluation
```
f1, precision, recall = my_model.evaluate(dev_data)
```

### Benchmark datasets
* Precision = 0.92
* Recall = 0.92
* F1 = 91.87

### jupyter file
* bert_ner.ipynb

### Demo video
 [video](https://youtu.be/Zu9q4jn-pJs)<br>

### Citation
```
@misc{devlin2018bert,
    title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
    author={Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
    year={2018},
    eprint={1810.04805},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
