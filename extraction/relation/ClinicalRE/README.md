# ClinicalRE
- Relation extraction from clinical texts using domain invariant convolutional neural network
- Sunil Kumar Sahu, Ashish Anand, Krishnadev Oruganty, Mahanandeeshwar Gattu: “Relation extraction from clinical texts using domain invariant convolutional neural network.” ACL 2016 (2016): 206.

## Original Code
https://github.com/sunilitggu/relation_extraction_in_clinical_text

## Description
- It used convolutional neural network to solve this problem. 
- The picture is the architecture of the model. It’s a typical CNN architecture. We first extracted the features of the words from the sentence, then embedded the features and sent them to convolution layer, then max-polling layer, fully-connected layer and softmax layer. And then we can get the most possible relation type of these two entities.
- We try to extract feature by using genia tagger, which is a tagger for biomedical text. Furthermore, we use dropout technique in output of max pooling layer and adam technique to optimize loss function.
- Use cross-validation to analyse the model.

<img src="/extraction/relation/ClinicalRE/image/appro.png">

## Input and Output
- Input and output for prediction  
	Because using cross-validation, the steps of training, prediction, and evaluation have been combined together.

- Input and output for training  
	* Input  
		Sentence e1 e1Type e1StartPosition e1EndPosition e2 e2Type e2StartPosition e2EndPosition RelationType
	* Output  
		Precision, Recall & F1 Score
		
	You can check the test folder for more information.

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

![eval](/extraction/relation/ClinicalRE/image/eval.png)

## Demo
- [Link to the Jupyter Notebook](/extraction/relation/ClinicalRE/code/ClinicalRE.ipynb)
- [Link to the video on Youtube](https://youtu.be/DO6raNbp0cg)