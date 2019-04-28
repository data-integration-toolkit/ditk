# Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing
Lee, J. and Seo, S. et al, **Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing.**, 2019, arXiv:1901.08163
This repository contains the source code for the Relation Extraction presented in the following research publication ([link](https://arxiv.org/pdf/1901.08163.pdf))

### Requirements<br>
* python 3.6
* tensorflow >= 1.6

### Input Data
* SemEval2010
* DDI2013
* NYT

### Data Preparation 
* SemEval2010 - Run SemEval2010_to_common function in SamEval2010_util.py
```
main():
  SemEval2010_to_common('./SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
* DDI2013 - Run readData function in ./DDI2013_all_data/DDI_util.py
```
inputfiles=["test/MedLine", "test/DrugBank"]
readData(inputfiles)
```
* NYT - Run nyt_to_common function in ./NYT_data/nyt_util.py
```
label = nyt_to_common('test.json', 1)
```

### Download pretrain model from [here](https://drive.google.com/open?id=1Trl1GQLWZn19LvelL-6clATvATKOPH77) and unzip the files in data directory.<br><br>


### main.py
* Create model
```
my_model = LSTM_relation_extraction()
```
* Load dataset
```
FLAGS.embeddings = 'glove300'
FLAGS.data_type = 'semeval2010'

files_dict = {
    "train": "data/"+FLAGS.data_type+"/trainfile.txt",
    "test": "data/"+FLAGS.data_type+"/testfile.txt",,
}
train_data, test_data = my_model.read_dataset(files_dict)
```
* Train model - Call save_model when training is finished. (The trained model saved into runs/FLAGS.data_type/checkpoint
```
my_model.train(train_data)
```
* Predict - Call load_model to load pretrained model from runs/FLAGS.data_type/checkpoint
```
predictions, output_file_path = my_model.predict(test_data)
```
* Evaluation
```
my_model.evaluate(None, predictions)
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

### Demo video
you can see jupyter file, [here](https://github.com/easy1one/ditk/blob/develop/extraction/named_entity/ner_with_ls/main_ver.ipynb)<br><br>

### Citation
```
@misc{lee2019semantic,
    title={Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing},
    author={Joohong Lee and Sangwoo Seo and Yong Suk Choi},
    year={2019},
    eprint={1901.08163},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
