# NFETC
Neural Fine-grained Entity Type Classification with Hierarchy-Aware Loss
Paper Published in NAACL 2018: [NFETC](https://arxiv.org/abs/1803.03378)
Original github for NFETC model: https://github.com/billy-inn/NFETC

### Prerequisites
- tensorflow >= r1.2
- hyperopt
- gensim
- sklearn
- pandas

### Dataset
Run `./download.sh` to download the corpus the pre-trained word embeddings
Available Dataset Name:
  1) **wiki**: Wiki/FIGER(GOLD) with original freebase-based hierarchy
  2) **ontonotes**: ONTONOTES
  3) **wikim**: Wiki/FIGER(GOLD) with improved hierarchy
you can add "Others" folder under "data" directory for other input which is not from Wiki/Onotnotes.

#### Note about wikim
Before preprocessing, you need to:
1. Create a folder `data/wikim` to store data for Wiki with the improved hierarchy
2. Run `python transform.py`



## To Run main.py
main.py call all functions for NFETC model; 5 things
**1. Create NFETC model**
**2. Read dataset**
**3. Train data**
**4. Predict data**
**5. evaluate data**
Lastly, print output file path.

### 1. Create NFETC model
main.py import NFETC class from **entity_tpying_subclass.py** and it is a subclass of **entity_typing.py** which is abstract class for entity typing group.

### 2. Read dataset
You **MUST** run `read_dataset` function for any functions(train/predict/evaluate).
#### 1) Input file
**(must be named imputation_test_input.tsv OR imputation_test_input.txt)**
Number of input for this model must be ONE;  raw data(.txt) or filtered data(.tsv)
raw data has ["p1", "p2", "Text", "Mention", "Types", "f"] format which is same as corpus format that dowloaded data by using  download.sh script in `data` folder.
But filtered data has ["p1", "p2", "Context", "Mention", "Types"]
Both are fine to run main.py and NFETC.preprocess_helper() will deal with preprocess for each case. 
One important thing is extension; one of filtered data should be .tsv abd one of raw data should be .txt
#### 2) options
NFETC needs 3 options; data_name/ ratio/ model_name/ (optional) epoch_num
**data_name**
- wiki
- ontonotes
- wikim
- others
- if your data is not from wiki/ontonotes/wikim(which are avaliable by download.sh) then put the input in `Others` folder and write others as data_name in main.py
**model_name**
- "best_nfetc_wiki": param_space_best_nfetc_wiki,
- "best_nfetc_wiki_hier": param_space_best_nfetc_wiki_hier,
- "best_nfetc_ontonotes": param_space_best_nfetc_ontonotes,
- "best_nfetc_ontonotes_hier": param_space_best_nfetc_ontonotes_hier,
**ratio**
- train/ dev/ test
- If you want to use your input data to train, (1.0, 0, 0)
- If you want to use your input data to evaluate, (0, 0, 1,0)
- deafult ratio is (0.7, 0.15, 0.15)
**epoch_num**
- number of epoches for your training, default value is 5

#### 3) Output
After preprocessing, myModel.read_dataset(file_path, options) will return dataset for training and testing, which will be used for train/predict/evaluate functinos.

### 3. train data
NFETC model will be train with train_data which is returned by read_data function
train_data = list(zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train))

### 4. Predict data
NFETC model will predict test_data which is returned by read_data function
test_data = list(zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train))
Then, it will return predict_data which has types for mentions
test_data and pred_data will be written in `output` folder

### 5. evaluate data
NFETC model will compute score using 3 metrics; Strict/ Macro F1/ Micro F1.
The scores for each run and the average scores are also recorded in one log file stored in folder `log`

## [Optional] Hyperparameter Tuning for models 
Run `python task.py -m <model_name> -d <data_name> -e <max_evals> -c <cv_runs>`
The searching procedurce is recorded in one log file stored in folder `log`


## Sample data with Sample word embedding is on the current repo
- There is Sample data() on `data/Other`

## Demo Video on YouTube
https://youtu.be/oNmLbJU-HHI
Run Sample input with only 1 epoch training NFETC model on Jupyter
Jupyter has same flow with main.py here.


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
