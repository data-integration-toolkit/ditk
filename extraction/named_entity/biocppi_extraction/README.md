## biocppi_extraction

- **Title of paper:** An end-to-end deep learning architecture for extracting protein-protein interactions affected by genetic mutations
- **Full Citation:** Tung,T. and Kavuluru,R. An end-to-end deep learning architecture for extracting protein-protein interactions affected by genetic mutations. Database(2018) Vol. 2018: article ID bay092; doi:10.1093/database/bay092

___



### Original code URL:

___

https://github.com/bionlproc/biocppi_extraction

[note that pre-trained word embeddings are provided by the author of the original code]

### Description

___

- biocppi_ectraction is a tool for named entity recognition
  - The main task is to identify which tokens [words] are part of a named entity. This tool only specifies miscellaneous entities [i.e. no entity 'types' are identified] and the architecture is shown below:

![image](https://drive.google.com/uc?export=view&id=1WvEa1Bx3-0PO6m9l0llSsfLtIqiDyTb-)

- The diagram shows the flow of the model from input to output. The model generates character level embeddings, and character type embeddings which are then passed through a convolutional neural network with 50 filters. Each filter is then squashed via maxed pooling, which generates 50 features for a given token. These features are then concatenated with pretrained word embeddings of 200 dimensions, as well as 32 more features semantically describing the word type. The 282 feature vector [per token] is then fed to a bidirectional LSTM, which is paired with a fully connected softmax output layer to identify a label for each token! Several tunable hyperparameters are also available to the user.



### Input and Output

___

The model is capable of generalizing the input of three different datasets, namely:

- **CHEMDNER |** **https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/**
- **CoNLL 2003** **|** **https://github.com/glample/tagger/tree/master/dataset**
- **OntoNotes 5.0** **|** **https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO**

The format of these datasets can be found in the above URLs. The model internally generates a 'ditk' format, in which each line has 15 items separated by spaces. Each sentence is denoted by a blank line in the input file. Of the 15 features, this model only uses two [token and label, shown below]

- Token _ _ Label _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

The model outputs predictions in a similar format, namely [per line, with sentences separated by a blank line]:

- Token true_label predicted_label

See `test/` folder for sample inputs and outputs.



### Evaluation

___

This model was evaluated on the three datasets listed above [CHEMDNER, CoNLL 2003, and OntoNotes 5.0]. The evaluation metrics reported on the test set for each of the datasets are:

- Precision = true_positives/(true_positives+false_positives)
- Recall = true_positives/(true_positives+false_negatives)
- F1 = (2*Precision*Recall)/(Precision+Recall)
- The results for each of the datasets is shown below:

![image](https://drive.google.com/uc?export=view&id=1Rj00I8YkL5mhicCXbsAzps57aj-dbOy8)

- Note: The PPIm dataset was the dataset the author originally trained and tested this model on. The results shown for that dataset are to indicate the author's results were reproducible with this model refactorization.



### Demo

___

Steps to get up and running:

1. Set up the python environment [**`linux` required**]:

   - Option a: install one thing at a time, and check along the way:

   ```python
   sudo apt-get install python-pip python-virtualenv
   virtualenv -p python2.7 biocppi
   source biocppi/bin/activate
   # install tensorflow
   pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0
   rc0-cp27-none-linux_x86_64.whl
   
   python -c 'import tensorflow'
   
   # install Fold
   pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1
   -cp27-none-linux_x86_64.whl
   
   python -c 'import tensorflow_fold'
   
   # pip install numpy
   pip install pandas
   pip install sklearn
   pip install nltk
   
   ```

   - Option b: use the requirements file:

   ```python
   sudo apt-get install python-pip python-virtualenv
   virtualenv -p python2.7 biocppi
   source biocppi/bin/activate
   
   pip install -r requirements.txt
   ```

2. Clone this repository [refer to Github clone documentation]

3. Run! Change around parameters in the `main.py` file or check out the jupyter notebook for examples

   - **REQUIRED:** open `main.py` and place your own datapaths in for
     - `embeddings_path` [**note that the PubMed-w2v.txt embeddings are available on the authors github linked at the top of this README**]
     - modify `dataset_name, dataset_dir` as well as `raw_*` filenames to point to your local copy of the datasets [datasets can be downloaded from links provided above]

   ```
   python main.py
   ```



Jupiter notebook:

`biocppi_extraction.ipynb`



Youtube video demonstration:

https://youtu.be/9E3-w1JSW1I



Note: slides documention much of the same information can be viewed here:

https://docs.google.com/presentation/d/1yNDu5t30yB5mbvlF-j2N3xAHa_O3S5BhaMHfyvFfLjE/edit?usp=sharing