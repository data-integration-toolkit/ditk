#BioBert
###Features

- **BioBERT: a pre-trained biomedical language representation model for biomedical text mining**
 Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, et al.
 
 - **Links**: 
 **github:** https://github.com/dmis-lab/biobert  
** paper:** https://arxiv.org/pdf/1901.08746.pdf
 **youtube video:** https://youtu.be/OEBIzOxOnNc
 **jupyter notebook:** https://colab.research.google.com/drive/11e61aUEKZrQC9Y4Z1F0okpvfDxDqv55H
 **final_presentation:** https://docs.google.com/presentation/d/1GZWzK_W9kk77VRRQF5R8A4FAKsZerK1p1txg5Ph2i7s/edit#slide=id.p1
 
 - **Please note that the original code is in python2 and I have converted the code to python3**
 
- **Input format follows the NER group's ditk format**, which consists of 17 columns:
  Word, POS Tag, Chunk, Entity Type, Document ID, part number, Word number, parse bit, predicate lemma, frameset ID, Word Sense, Speaker/Author, Named Entity, Predicate Args, Predicate Args, Conference
  
**Please note, my project only uses 2 of the above cols - Word and Entity Type,** though all columns need to be fed to the model, with '-' wherever value is not needed.
- **Output format** is again **standard** for the entire **group**:
   **WORD, True type, predicted type.**

- Input/Output format for training and testing/predicting is the same.

- The **project uses pretrained BERT models** to perform NER. The models have been trained on PUBMED and PMC abstracts. The pre-trained model has been **stored on my Google storage as a bucket** and the in code API calls access it live to build the biomed model. The BERT architecture uses bi-directional transformers to give state of the art results.


- **How to run the code:**
	Well the jupyter notebook shows clearly how to clone the ditk repo and run my code. However just to be sure:
	**First make sure you are in the right directory. That is change the directory to ditk/extraction/named_entity/biobert**
	Then as per the TA's suggestion, we can run the code by simply:
		python3 main.py 'tests/sample_input.txt'
	please note that this code runs in the aforementioned way only. Trying to run it by invoking the following will **throw ERRORS**:
		python3 -m biobert.main 'sample_input.txt'
	 **Also note supply the full path of the text file, if the text file is not placed inside biobert directory.**
	 Also **ensure that pip3 install -r requirements.txt has been run to satisfy the package dependencies.**

- **Benchmark datasets**:  CoNLL 2003, OntoNotes 5.0, CHEMDNER dataset

- **Evaluation metrics:**
   precision, recall, f1
   
precision values for full datasets run for over 12 hours:
| Dataset  |  Precision |
| :------------ | :------------ |
|Conll 2003 | 81.97  |
|Ontonotes 5.0   |  78.53 |
|CHEMDNER BC4 | 89.96 |

