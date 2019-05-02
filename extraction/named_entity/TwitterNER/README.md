# TwitterNER - Semi-Supervised NER on Noisy Text

### Full Citation
Shubhanshu Mishra and Jana Diesner. Semi-Supervised Named entity recognition in Noisy Text. Proceedings of Noisy User-generated Text(NUT) at International Conference on Computational Linguistics (COLING), Osaka, Japan, 2016. https://aclweb.org/anthology/W16-3927

### Input/Output format for Prediction
Input: Format similar to Ner_test_input.txt present in NoisyNLP/tests
Output: Word, Actual Tag, Predicted Tag

### Input format for Training
Word, Actual Tag

### Working
Convert BIO Labels to BIEOU format Gather all the features using Feature Extractors. Train on CRFs using SGD with L2 norm. Update the unsupervised features using the new batch of un-labelled test data, and then retrain our model on the original training data. Regex Features generated on a per-token level and pairs multiplied. Gazetteers generated on a per-token level using window-size to determine interaction. Word Clusters generated in an unsupervised manner using Brown and Clark clustering. Global Features are the average values of word representation & binary presence of dictionary and cluster features.
![Image description](https://github.com/napsternxg/TwitterNER/blob/master/COLING2016-WNUT-Model-Architechture.png)

### Installation & Working
Install all the requirements given and then add the glove vectors needed for training. 

```sh
pip install -r requirements.txt
cd data
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
cd ..
```

* All the code created as part of the re-factored module is present in TwitterNER/twitterner.py and can be run using TwitterNER/main.py. 
* It follows the parent class template which is seen in ner.py. 
* All the models for the benchmark datasets can be found in TwitterNER/models folder.
* All the datasets can be found in the TwitterNER/data folder. 

### Evaluation Datasets
![Image description](https://github.com/KhadijaZavery/ditk/blob/develop-py2/extraction/named_entity/TwitterNER/datasets.png)

### Evaluation Metrics
1. Precision
2. Recall
3. F1- Score

### Further Links
Jupyter Notebook: 
Youtube Video: 
Auto-generated Documentation: https://dbaumgarten.github.io/FToDTF/
Architecture Documentation: https://github.com/dbaumgarten/FToDTF/blob/master/docs/architecture/architecture.md
