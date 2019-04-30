# Text Similarity Encoding

- Similarity encoding for learning with dirty categorical variables
- Patricio C., Gaël V., Balázs K. (2018). Similarity encoding for learning with dirty categorical variables. Manuscript accepted for publication at: Machine Learning Journal. Springe.

## Original Code

- Github: https://github.com/pcerda/ecml-pkdd-2018

## Description

- Similarity Encoding is a method aim to encode text(string) by its similarity to other text(string) in the corpus. Similarity Encoding models the embeddings of text by using different similarity measuring functions such as "Jaro-Winkler", "Levenshtein-ratio", "3-gram" and more. 
- The Similarity Encoding model provided in this repository contains some main functions such as embedding training data and predicting embeddings using training data. It also provides supporting functions such as finding embedded similarities of two string, and also evaluating the embeddings by finding a similarity value and compare to the original similarity value.

## Input and Output

Input for embedding:

​	Files:

   - train_input (this is the training data that will be embedded)
   - predict_input (this is the data that need to be embedding using training data)
   - similarity_input1 & similarity_input2 (only if using predict_similarity, they are the two strings for finding similarity)

​	Parameters:

- dimension: dimension to embed each entity and relationship to

Output:

Files:

- train_output.txt (embeddings are outputted in a format of <string, vector of dimension>)
- predict_output.txt (embeddings are outputted in a format of <string, vector of dimension>)

## Evaluation

- Benchmark datasets:

  - Employee Salary

  - SICK

  - Semi2017

    |                 | Normalized average error<br />(predicted_similarity - actual_similarity)/N |
    | --------------- | ------------------------------------------------------------ |
    | Employee Salary | 0.126                                                        |
    | SICK            | 0.648                                                        |
    | Semi2017        | 0.375                                                        |

    

## Demo

- Link to the Jupyter Notebook
- Link to the video on Youtube