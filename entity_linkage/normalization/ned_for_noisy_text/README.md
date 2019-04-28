# Method Name
- Named Entity Disambiguation for Noisy Text
- Yotam Eshel, Noam Cohen Kira Radinsky, Shaul Markovitch, Ikuya Yamada and Omer Levy. Accepted to CoNLL, 2017. <https://arxiv.org/pdf/1706.09147.pdf>

## Original Code
<https://github.com/yotam-happy/NEDforNoisyText>

## Description
- We present WikilinksNED, a large-scale NED dataset of text fragments from the web, which is significantly noisier and more challenging than existing newsbased datasets. To capture the limited and noisy local context surrounding each mention, we design a neural model and train it with a novel method for sampling informative negative examples. We also describe a new way of initializing word and entity embeddings that significantly improves performance. Our model significantly outperforms existing state-ofthe-art methods on WikilinksNED while achieving comparable performance on a
  smaller newswire dataset.

- ```Model.png
  ![Model]
  ```

## Input and Output
- Articles, a list of words with its Name Entity Tag
- Entity url

## Evalution
- CoNLL 2003
- P@1 (70.2%)

## Demo
- Link to the Jupyter Notebook 
- Link to the video on Youtube
