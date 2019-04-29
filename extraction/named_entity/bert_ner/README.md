# NER with BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Devlin, Jacob et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**, CoRR abs/1810.04805, 2018<br>
This repository contains the source code for the NER system presented in the following research publication ([link](https://arxiv.org/pdf/1810.04805.pdf))

### Requirements<br>
* python 3.6
* tensorflow >= 1.6

### Input Data
* CoNLL2003
* OntoNotes2012

### Input format for training/prediction
* The input format is generalized for the whole NER group
* It contains multiple columns got from CoNLL2003 and CoNLL 2012 (separated by space)
* Sample
* ![Common Input](CommonInput.png)
```
Yes UH (TOP(S(INTJ*) O bc/cnn/00/cnn_0003 0 0 - - - Linda_Hamilton * -
they PRP (NP*) O bc/cnn/00/cnn_0003 0 1 - - - Linda_Hamilton * (15)
did VBD (VP*) O bc/cnn/00/cnn_0003 0 2 do 01 - Linda_Hamilton (V*) -
```
* Common output
```
word entity predicted_entity
```

### Sample Test data
* ./textexample

### Download pretrain model
* Download pretrain bert model from [here](https://drive.google.com/open?id=1UBgb9OlLFvYGzpUufaj9Voe36muxW4Ga) and unzip the file in bert_ner directory.


### How to run test
```
python sample_test.py
```

### Citation
```
@misc{devlin2018bert,
    title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
    author={Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
    year={2018},
    eprint={1810.04805},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
