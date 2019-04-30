# NFETC
Neural Fine-grained Entity Type Classification with Hierarchy-Aware Loss<br>
Paper Published in NAACL 2018: [NFETC](https://arxiv.org/abs/1803.03378)<br>
Original github for [NFETC model](https://github.com/billy-inn/NFETC)<br>

### Prerequisites
- tensorflow >= r1.2
- hyperopt
- gensim
- sklearn
- pandas

### Dataset
Run `./download.sh` to download the corpus the pre-trained word embeddings<br>
Available Dataset Name:<br>
  1) **wiki**: Wiki/FIGER(GOLD) with original freebase-based hierarchy
  2) **ontonotes**: ONTONOTES
  3) **wikim**: Wiki/FIGER(GOLD) with improved hierarchy
you can add "Others" folder under "data" directory for other input which is not from Wiki/Onotnotes.<br>

#### Note about wikim
Before preprocessing, you need to:<br>
1. Create a folder `data/wikim` to store data for Wiki with the improved hierarchy<br>
2. Run `python transform.py`<br>


## To Run main.py
main.py call all functions for NFETC model; 5 things<br>
**1. Create NFETC model**<br>
**2. Read dataset**<br>
**3. Train data**<br>
**4. Predict data**<br>
**5. evaluate data**<br>
Lastly, print output file path.

### 1. Create NFETC model
main.py import NFETC class from **entity_tpying_subclass.py** and it is a subclass of **entity_typing.py** which is abstract class for entity typing group.

### 2. Read dataset
You **MUST** run `read_dataset` function for any functions(train/predict/evaluate).<br>
#### 1) Input file
**(must be named imputation_test_input.tsv OR imputation_test_input.txt)**<br>
Number of inputs for this model must be ONE of two;  raw data(.txt) or filtered data(.tsv)<br>
raw data has ["p1", "p2", "Text", "Types", "f"] format which is same as corpus format that dowloaded data by using  download.sh script in `data` folder.<br>
But filtered data has ["p1", "p2", "Context", "Mention", "Types"]<br>
Both are fine to run main.py and NFETC.preprocess_helper() will deal with preprocess for each case. <br>
One important thing is extension; one of filtered data should be .tsv abd one of raw data should be .txt<br>
#### 2) options
NFETC needs 3 options; data_name/ ratio/ model_name/ (optional) epoch_num<br>
**data_name**<br>
- wiki<br>
- ontonotes<br>
- wikim<br>
- others<br>
- if your data is not from wiki/ontonotes/wikim(which are avaliable by download.sh) then put the input in `Others` folder and write others as data_name in main.py<br>
**model_name**<br>
- "best_nfetc_wiki": param_space_best_nfetc_wiki,<br>
- "best_nfetc_wiki_hier": param_space_best_nfetc_wiki_hier,<br>
- "best_nfetc_ontonotes": param_space_best_nfetc_ontonotes,<br>
- "best_nfetc_ontonotes_hier": param_space_best_nfetc_ontonotes_hier,<br>
**ratio**<br>
- train/ dev/ test<br>
- If you want to use your input data to train, (1.0, 0, 0)<br>
- If you want to use your input data to evaluate, (0, 0, 1,0)<br>
- deafult ratio is (0.7, 0.15, 0.15)<br>
**epoch_num**<br>
- number of epoches for your training, default value is 5<br>

#### 3) Output
After preprocessing, myModel.read_dataset(file_path, options) will return dataset for training and testing, which will be used for train/predict/evaluate functinos.<br>

### 3. train data
NFETC model will be train with train_data which is returned by read_data function<br>
train_data = list(zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train))<br>

### 4. Predict data
NFETC model will predict test_data which is returned by read_data function<br>
test_data = list(zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train))<br>
Then, it will return predict_data which has types for mentions<br>
test_data and pred_data will be written in `output` folder<br>

### 5. evaluate data
NFETC model will compute score using 3 metrics; Strict/ Macro F1/ Micro F1.<br>
The scores for each run and the average scores are also recorded in one log file stored in folder `log`<br>

## [Optional] Hyperparameter Tuning for models 
Run `python task.py -m <model_name> -d <data_name> -e <max_evals> -c <cv_runs>`<br>
The searching procedurce is recorded in one log file stored in folder `log`<br>


## Sample data with Sample word embedding is on the current repo
Sample input data on Others folder; `data/Other/entity_typing_test_input.txt`<br>
Extracted downsized word embedding only for sample data is under data folder; `data/glove.840B.300d.txt`<br>
*Therefore, you can directly run main.py to call ALL functions with sample data and word embedding dataset that I uploaded!*<br>

## Benchmark datasets
- FIGER(GOLD)
- OntoNotes

## Evaluation matrics and results
- (Strict Acc, Macro F1, Micro F1)
- (0.543, 0.717, 0.649) on OntoNotes benchmark

## Demo Video on YouTube
https://youtu.be/u5q_VIAOy90 <br>
Run Sample input with only 1 epoch training NFETC model on Jupyter<br>
**Jupyter Link** is [here](https://github.com/easy1one/ditk/blob/develop/entity_linkage/typing/nfetc/main_ver.ipynb) <br>


### Cite
If you found this codebase or our work useful, please cite:
```
@InProceddings{xu2018neural,
  author = {Xu, Peng and Barbosa, Denilson},
  title = {Neural Fine-Grained Entity Type Classification with Hierarchy-Aware Loss},
  booktitle = {The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2018)},
  month = {June},
  year = {2018},
  publisher = {ACL}
}
```
