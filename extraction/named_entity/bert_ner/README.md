# NER with BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Devlin, Jacob et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**, CoRR abs/1810.04805, 2018<br>
This repository contains the source code for the NER system presented in the following research publication ([link](https://arxiv.org/pdf/1810.04805.pdf))

### Requirements<br>
* python 3.6
* tensorflow >= 1.6

### Input Data
* CoNLL2003
* OntoNotes2012

### Data Preparation 
* CoNLL2003 - Run conll2003_to_ditk function in convert_to_2003.py
```
main():
  lines = [line.strip() for line in open('./conll2003_NERdata/train.txt')]
  ret = conll2003_to_ditk(lines)
  with open('./NERdata/dev_test.txt', 'w') as f:
    for line in ret:
      f.writelines(' ".join(line)+'\n')
```
* OntoNotes2012 - 

### Download pretrain model from [here](https://drive.google.com/open?id=1Trl1GQLWZn19LvelL-6clATvATKOPH77) and unzip the files in data directory.<br><br>


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

### Demo video
you can see jupyter file, [here](https://github.com/easy1one/ditk/blob/develop/extraction/named_entity/ner_with_ls/main_ver.ipynb)<br><br>

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
