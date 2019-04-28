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
* SemEval2010 - Run SemEval2010_to_common function in SamEval2010_util.py
```
main():
  SemEval2010_to_common('./SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
```
* DDI2013 - Run readData function in ./DDI2013_all_data/DDI_util.py
```
inputfiles=["test/MedLine", "test/DrugBank"]
readData(inputfiles)
```
* Download data from [here](https://drive.google.com/open?id=1-t7gmA3cxrz3ybAfO5gPFYZxWEAzS4E7)
* NYT - Run nyt_to_common function in ./NYT_data/nyt_util.py
```
label = nyt_to_common('test.json', 1)
```

### Download pretrain model from [here](https://drive.google.com/open?id=1ZjmlWFsMS86ftdxrWdIuUAuG3Cmw9isq) and unzip the files in runs /pretrained_file_name/ directory.<br>


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

### jupyter file
* lstm_relation_extraction_with_semeval2010_and_glove300.ipynb

### Demo video
*  [video]()<br>

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
