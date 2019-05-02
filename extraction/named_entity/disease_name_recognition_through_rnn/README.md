## disease_name_recognition_through_rnn

- **Title of paper:** Recurrent neural network models for disease name recognition using
  domain invariant features
- **Full Citation:** Sunil Kumar Sahu,et. al. Recurrent neural network models for disease name recognition using domain invariant features. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pages 2216–2225, 2016

___



### Original code URL:

___

<https://github.com/sunilitggu/disease_name_recognition_through_rnn>

### Description

___

- disease_name_recognition_through_rnn
  - The main task is to identify which tokens [words] are part of a named entity. This tool only specifies miscellaneous entities [i.e. no entity 'types' are identified] and the architecture is shown below:

![image](https://drive.google.com/uc?export=view&id=1tGFLFTW_yhiLr9EJTOR6qaVHZU7dftAO)

- The diagram shows the flow of the model from input to output. The model generates character level embeddings which are then concatenated with learned word embeddings. The concatenated feature vectors [one per token] are fed to a bidirectional LSTM, which is paired with a dense output layer to identify a label for each token! Several tunable hyperparameters are also available to the user.



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

![image](https://drive.google.com/uc?export=view&id=1mqirDxPS_nuHg_0pJ7Z1m0WXQKbNmX-Y)

- Note: The NCBI dataset was the dataset the author originally trained and tested this model on. The results shown for that dataset are to indicate the author's results were reproducible with this model refactorization.



### Demo

___

Steps to get up and running:

1. Set up the python environment:

   ```python
   # create env
   conda create -n drtrnn python=2.7 pip
   
   # activate env
   source activate drtrnn
   
   # install requirements
   while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt 2>env_error.log
   
   ```

2. Clone this repository [refer to Github clone documentation]

3. Run! Change around parameters in the `main.py` file or check out the jupyter notebook for examples

   - **REQUIRED:** open `main.py` and place your own datapaths in for
     - modify `dataset_name, dataset_dir` as well as `raw_*` filenames to point to your local copy of the datasets [datasets can be downloaded from links provided above]

   ```
   python main.py
   ```



Jupiter notebook:

`disease_name_recognition_through_rnn.ipynb`



Youtube video demonstration:

link here...



Note: slides documention much of the same information can be viewed here:

https://docs.google.com/presentation/d/1Dn8SnCtbDlCMFMoC4dvD_QUyfFb_pkxLs8RfhXLmRjc/edit?usp=sharing