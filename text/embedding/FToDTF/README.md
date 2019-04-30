# FToDTF - FastText on Distributed TensorFlow
This software is an implementation of FastText on Distributed Tensorflow. This project is a re-factored version of an already impelemented project present at https://github.com/dbaumgarten/FToDTF.  
FToDTF uses unsupervised machine-learning algorithm to calculate vector-representation of words. These vector representations can then be used for things like computing the similarity of words to each other or association-rules (e.g. paris is to france like X to germany).
In contrast to the original implementation of fasttext (https://github.com/facebookresearch/fastText) this implementation can use GPUs to accelerate the training and the training can be distributed across multiple nodes.


### Full Citation
Based on Research Paper: Piotr Bojanowski, Edouard Grave, Armand Joulin and Tomas Mikolov's "Enriching word vectors with Subword Information". (https://arxiv.org/pdf/1607.04606v2.pdf)
Published in Transactions of the Association for Computational Linguistics, Volume 5, July 15th 2016.

### Input/Output format for Prediction
Input: Word(s)/Sentences(s)/Paragraph(s)
Output: Word/Sentence Embeddings

### Input format for Training
{Word/Sentence, Word/Sentence, Similarity score, Label-Test/Train*}
\*- Optional

### Working
Each word represented as a bag of character n-grams with special boundary symbols < and >. Associate a vector representation zg to each n-gram g.  Represent a word by the sum of the vector representations of its n-grams. Use a hashing function to bound memory requirements that maps n-grams to integers in 1 to K. Ultimately, a word is represented by its index in the word dictionary and the set of hashed n-grams it contains - Fowler-Noll-Vo hashing function. The scoring function – 

![alt text] (https://github.com/KhadijaZavery/ditk/blob/develop/text/embedding/FToDTF/FToDTF%20Architecture.png)

### Installation & Working
Requires Python 3.5+ to run. Clone the repository and use this command to install the requirements.
```sh
python setup.py install
```
* All the code created as part of the re-factored module is present in FToDTF/fasttext.py and can be run using FToDTF/main.py. 
* It follows the parent class template in FToDTF/text_embedding.py. 
* All the models for the benchmark datasets can be found in FToDTF/ models folder.
* All the datasets can be found in the FToDTF/data folder. 
* The variable 'modelpath' in the code is the path where you want the checkpoints of previously trained model can be saved and can be modified based on user's requirement.  


### Evaluation Datasets
1.	WordSimilarity-353 Test Collection contains two sets of English word pairs along with human-assigned similarity judgements.
2.	SICK 2014(SemEval 2014, Task 1)
10,000 English sentence pairs, generated starting from two existing sets: the 8K ImageFlickr data set and the SemEval 2012 STS MSR-Video Description data set.
3.	SemEval 2017, Task 1
Semantic Comparison of Words and Texts for Semantic Textual Similarity.
4.   SemEval 2014, Task 10
2,500 English sentences annotated with relatedness in meaning. 

### Evaluation Metrics
1. Pearson Relation
2. Mean Square Error 
3. Spearman Correlation

### Further Links
Jupyter Notebook: 
Youtube Video: 
Auto-generated Documentation: https://dbaumgarten.github.io/FToDTF/
Architecture Documentation: https://github.com/dbaumgarten/FToDTF/blob/master/docs/architecture/architecture.md
