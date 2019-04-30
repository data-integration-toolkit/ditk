Robust Lexical Features for Improved Neural Network Named-Entity Recognition
================================================================
Abbas Ghaddar and Philippe Langlais , **Robust Lexical Features for Improved Neural Network Named-Entity Recognition**, In Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018) <br>
This repository contains the source code for the NER system presented in the following research publication ([link](http://aclweb.org/anthology/C18-1161))

## Requirements<br>
* python 3.6
* tensorflow>=1.6
* pyhocon (for parsing the configurations)


## Prepare the Input data<br>
    Input must be "conll2003" or "ontonotes" to run main.py file because pre-trained "conll.joblib" and "ontonotes.joblib" are needed to run the model. This is not just for word-embedding. They applied 4 embeddings to get feature representation; Word-embedding, Character, Capitalized, and Lexcial Similarity features. I could not find the script to create pre-trained embedding for this paper. Therefore, main.py will work with subset of conll (with "conll" as dataset_name) or ontonotes (with "ontonotes" as dataset_name). 
    1)Input: [word, pos, chunk, entity_tag] format is needed to run the code. Since it needs pre-trained embeddings for input words, conll2003 and ontonotes5.0 are pre-trained respectively. (So I cannot train with any other words as inputs)
    2) Output: [word, truth_lable, predict_label]
#### Download pre-trained embedding data from [here](https://drive.google.com/open?id=1Trl1GQLWZn19LvelL-6clATvATKOPH77) and unzip the files in data directory.<br><br>
  
  
  
## To run main.py<br>
### 1. Create my model<br>
``` myModel = NER_with_LS(dataset_name)```
After creating my model, I will split the input files to "train", "dev", "test" with the ratio that users can modify in ```main.py```. It is an optional part but if you want to get all outputs according to your input file, then you should change the ratio to (0.0, 0.0, 1.0) because the default ratio is (7.0, 0.15, 0.15).<br>

### 2. Read dataset<br>
```data = myModel.read_dataset(file_dict)```
data var is dictionary to have {"train", "dev", "test"} data resepectively. <br>

### 3. Train the model
```myModel.train(data)```
It will train with data["train"] and data["dev"] and then save trained model.<br>

### 4. Save/Load model
```myModel.save_model()```
```myModel.load_model()```

### 5. Predict data 
```pred_labels = myModel.predict(data["test"])```
pred_labels are labels predicted by pre-trained model(or inital model) with data["test"]. So its size is equals to len(data["test"])<br>

### 6. Get ground truth lables
```ground_truth_labels = myModel.convert_ground_truth(data["test"])```
It will have same length with **pred_labels**<br>

### 7. Finally get Score 
```scores = myModel.evaluate(pred_labels, ground_truth_labels)```
Evaluateion matrics are "precision", "recall", "f1"<br><br><br>


## Benchmark datasets
* Conll 2003
* Ontonotes 5.0

## Evaluation metrics and results
* Precision, Recall, F1 Score
* 0.9125	0.9219	0.9172	(conll 2003)
* 0.4133	0.3974	0.4052  (ontonotes 5.0) with only trial test data (Not full dataset)

## Demo Video
https://youtu.be/O2pOc4Yl4R4 <br>
you can see jupyter file, [here](https://github.com/easy1one/ditk/blob/develop/extraction/named_entity/ner_with_ls/main_ver.ipynb)<br><br>

## Citation
Please cite the following paper when using our code: 

```
@InProceedings{ghaddar2018coling,
  title={Robust Lexical Features for Improved Neural Network Named-Entity Recognition},
  author={Ghaddar, Abbas	and Langlais, Phillippe},
  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
  pages     = {1896--1907},
  year      = {2018}
}

```
