
# Semantic Neural Network

## Title
Measuring Semantic Similarity Between Sentences Using a Siamese Neural Network.
## Citation
Alexandre Yukio Ichida, Felipe Meneguzzi, Duncan D. Ruiz.
2018 International Joint Conference on Neural Networks (IJCNN).
Paper:- http://www.meneguzzi.eu/felipe/pubs/ijcnn-sem-similarity-2018.pdf
## Prediction Format
**Input** - 2 sentences.<br />
**Output** - Similarity score (0-1).

 *Provide the tokenizer and model obtained from training in the implementation module.*
## Training Format
**Input:** 
* Three input files - *.txt or .csv* files for training, validation and testing.
* Word2Vec - Negative Sampling: https://code.google.com/archive/p/word2vec/

**Output:** 
* Model will be stored in .h5 format in the *basedir/models/*  folder.
* Tokenizer for the data trained will be stored in *basedir/tokenizers/*  folder.

## Approach
### Overview
This model uses a siamese GRU neural network architecture to measure the semantic similarity between two sentences. GRU (Gated Recurrent Units) is a recurrent network
architecture proposed to deal with long sequences, using a gating mechanism to create a memory control of values processed over time.

### Steps
This section describes in two steps the architecture of the neural network used to obtain the semantic similarity between two sentences.
1. Data Pre-processing Step
    * The pre-processing step consists of creating a numerical representation to a sentence by converting each word into an integer number.
    * For such conversion, we create a word dictionary of the corpus vocabulary and associate a unique numerical index for each word seen in the dataset.
    * We maintain word order from sentences in the resulting vector, preserving the original context and meaning of the sentence semantics.

2. Siamese GRU Model
    * Word embeddings: The input layer of our architecture converts each vector of indexes received from data pre-processing into a word distributed representation using Word2Vec Skip-Gram model.
    * Sentence representation: We make use of recurrent network architecture to process the word embedding sequences. It consists of two symmetric recurrent neural networks with shared weights to learn semantic differences between sentences.
    * Output layer: Learns a distance function- Manhattan Distance function, which results in a similarity metric between two encoded sentences.

## Steps to run the Model
**main.py: Entire model is triggered**
   * Provide Train, Dev and Test datafile in .txt format (s1,s2,relatedness_score)
   * Returns similarity_score for Test file in output directory.
   
**semantic_neural_network.py: Implementation for TextSemanticSimilarity abstract superclass**
   * read_dataset() - Supports SICK, SemEval2014, SemEval2017 and Generic format(s1,s2,relatedness_Score).
   * get_embedding_matrix() - Creates embedding matrix and tokenizers for input data.
   * train() - Train the model and creates .h5 file in results directory.
   * evaluate() - Predicts the similarity_scorre for Test file and store it in output directory. Computes the Pearson correlation co-efficient.
   * predict () - Reads the model file and tokenizer created in train() function and outputs the similarity score for the two sentences as input.
     
   
## Benchmark Datasets
1. The SICK dataset 
    * 10000 English sentence pairs extracted from the ImageFlick dataset2 and SemEval2012 semantic textual similarity video description data set.
    * 5000 sentence pairs as training set, 500 as validation set, and 4500 as test set.
 
2. SemEval 2014 dataset.
    *  Subset of the Linking-Tweets-to-News dataset,
    *  Subset of news article data in the DEFT DARPA project,
    *  Headlines mined from several new sources,
    *  Sense definitions from WordNet and OntoNotes.
 
3. SemEval 2017 dataset.
    * 5700 sentence pair for training, 1300 as validation set and 1500 as test set.

## Evaluation Metrics
**Pearson correlation-** Indicates the extent to which two variables are linearly related.

## Results obtained
| Dataset  | Pearson correlation|
| ------------- | ------------- |
| SICK  | 0.80684  |
| SemEval 2014  | 0.72789  |
| SemEval 2017  | 0.76723  |

## Jupyter notebook

https://colab.research.google.com/drive/1KHh6lo0PQZeja3eigCRJ9CFNuhCFz_dY

## YouTube links
https://www.youtube.com/watch?v=b1h8ATy7i5o&feature=youtu.be
