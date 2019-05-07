#Cross View Training
###Features

- **Semi-Supervised Sequence Modeling with Cross-View Training**
 Kevin Clark, Quoc V. Le, et al., EMNLP 2018, Brussels, Belgium
  - **Links**: 
 **github:** https://github.com/tensorflow/models/tree/master/research/cvt_text
 **paper:** https://arxiv.org/pdf/1809.08370.pdf
 **youtube video:** https://youtu.be/OEBIzOxOnNc
 **jupyter notebook:** https://colab.research.google.com/drive/15b1nK8lPh8WBQUdxat5V1CZrEw1CjkKM
 **final_presentation:** https://docs.google.com/presentation/d/1GZWzK_W9kk77VRRQF5R8A4FAKsZerK1p1txg5Ph2i7s/edit#slide=id.p1
 
 - **Please note that the original code is in python2 and I have converted the code to python3**
- **Input format follows the NER group's ditk format**, which consists of 17 columns:
  Word, POS Tag, Chunk, Entity Type, Document ID, part number, Word number, parse bit, predicate lemma, frameset ID, Word Sense, Speaker/Author, Named Entity, Predicate Args, Predicate Args, Conference
  
**Please note, my project only uses 2 of the above cols - Word and Entity Type,** though all columns need to be fed to the model, with '-' wherever value is not needed.
- **Output format** is again **standard** for the entire **group**:
   **WORD, True type, predicted type.**

- Input/Output format for training and testing/predicting is the same.

- The **model uses both supervised and unsupervised learning models** to perform NER. Supervised Learning is done with the help of labelled text, unsupervised learning is done by the auxiliary modules. **GloVe Embeddings and L1mb corpus** is required. A micro version of both have been stored in the codebase.

- **How to run the code:**
	Well the jupyter notebook shows clearly how to clone the ditk repo and run my code. However just to be sure:
	First make sure you are in the right directory. As per the TA's suggestion, we can run the code by simply:
		python3 main.py 'tests/sample_input.txt'
	please note that this code runs in the aforementioned way only. Trying to run it by invoking the following will **throw ERRORS**:
		python3 -m cvt.main 'sample_input.txt'
	 Also note **supply the full path of the text file, if the text file is not placed inside cvt directory.**
	 Also **ensure that pip3 install -r requirements.txt has been run** to satisfy the package dependencies.

- **Benchmark datasets**: CoNLL 2000, CoNLL 2003, OntoNotes 5.0, CHEMDNER dataset

- **Evaluation metrics:**
   precision, recall, f1
   
   precision values for full datasets run for over 12 hours:
| Dataset  |  Precision |
| :------------ | :------------ |
|Conll 2000   |  91.33 |
|Conll 2003 | 91.33  |
|Ontonotes 5.0   |  85.16 |
|CHEMDNER BC4 | 72.24 |

