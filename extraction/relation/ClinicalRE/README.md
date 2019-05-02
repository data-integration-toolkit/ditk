# ClincalRE
- Relation extraction from clinical texts using domain invariant convolutional neural network
- Sunil Kumar Sahu, Ashish Anand, Krishnadev Oruganty, Mahanandeeshwar Gattu: “Relation extraction from clinical texts using domain invariant convolutional neural network.” ACL 2016 (2016): 206.

## Original Code
https://github.com/sunilitggu/relation_extraction_in_clinical_text

## Description
- It used convolutional neural network to solve this problem. The picture is the architecture of the model. It’s a typical CNN architecture. We first extracted the features of the words from the sentence, then embedded the features and sent them to convolution layer, then max-polling layer, fully-connected layer and softmax layer. And then we can get the most possible relation type of these two entities. We try to extract feature by using genia tagger, which is a tagger for biomedical text. Furthermore, we use dropout technique in output of max pooling layer and adam technique to optimize loss function.
- A figure describing the model if possible

## Input and Output
- Input and output for Training
	-Input
	-Output

## Evalution
### Evaluation Datasets
* i2b2 2010
* CoNLL 2003
* OntoNotes 5.0
* CHEMDNER

### Evaluation Metrics
* Precision
* Recall
* F1 Score

### Evaluation Results

|#|i2b2 2010|SemEval 2010|DDI 2013|
|---|---|---|---|
|Precision|76.32%|18.22%|54.87%|
|Recall|67.24%|25.31%|43.52%|
|F1 Score|71.49%|21.27%|48.54%|

## Demo
- [Link to the Jupyter Notebook](#https://github.com/Candelaa/ditk/blob/develop/extraction/relation/ClinicalRE/code/ClinicalRE.ipynb)
- [Link to the video on Youtube](#https://youtu.be/DO6raNbp0cg)