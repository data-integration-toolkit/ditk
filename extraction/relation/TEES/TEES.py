import TEES
from TEES.Detectors.Preprocessor import Preprocessor
import sys
import os

class TEES():

    def __init__(self):
        pass

    def read_dataset(self, dataset_name="nyt"):  
        """
        Reads a dataset to be used for training
         
         Note: The child file of each member overrides this function to read dataset 
         according to their data format.
         
        Args:
            input_file: Filepath with list of files to be read
        Returns: 
            (optional):Data from file
        """
        pass
        
    def data_preprocess(self,input_data, output_file_name="output.xml", *args, **kwargs):
        """
         (Optional): For members who do not need preprocessing. example: .pkl files 
         A common function for a set of data cleaning techniques such as lemmatization, count vectorizer and so forth.
        Args: 
            input_data: Raw data to tokenize
        Returns:
            Formatted data for further use.
        """
	current_dir = os.getcwd()
        steps = "LOAD,BLLIP_BIO,STANFORD_CONVERT,SPLIT_NAMES,FIND_HEADS,SAVE"
        parseName = "McCC"
        requireEntities = False
        preprocessor = Preprocessor(steps, parseName, requireEntities)
        preprocessor.setArgForAllSteps("debug", False)
	preprocessor.process(input_data, os.getcwd()+"/"+output_file_name, model=None, logPath="AUTO")


    def tokenize(self, input_data ,ngram_size=None, *args, **kwargs):  
        """
        Tokenizes dataset using Stanford Core NLP(Server/API)
        Args:
            input_data: str or [str] : data to tokenize
            ngram_size: mention the size of the token combinations, default to None
        Returns:
            tokenized version of data
        """
        pass


    def train(self, train_file, devel_file, test_file, output):  
        """
        Trains a model on the given training data
        
         Note: The child file of each member overrides this function to train data 
         according to their algorithm.
         
        Args:
            train_data: post-processed data to be trained.
        
        Returns: 
            (Optional) : trained model in applicable formats.
             None: if the model is stored internally.
        """

    def predict(self, encoder="pcnn", selector="ave", dataset_name = "nyt"):   
        """
        Predict on the trained model using test data
        Args:
              entity_1, entity_2: for some models, given an entity, give the relation most suitable 
            test_data: test the model and predict the result.
            trained_model: the trained model from the method - def train().
                          None if store trained model internally.
        Returns:
              probablities: which relation is more probable given entity1, entity2 
                  or 
            relation: [tuple], list of tuples. (Eg - Entity 1, Relation, Entity 2) or in other format 
        """

    def evaluate(self, encoder="pcnn", selector="ave", dataset_name = "nyt"):
        """
        Evaluates the result based on the benchmark dataset and the evauation metrics  [Precision,Recall,F1, or others...]
         Args:
             input_data: benchmark dataset/evaluation data
             trained_model: trained model or None if stored internally 
        Returns:
            performance metrics: tuple with (p,r,f1) or similar...
        """

if __name__ == '__main__':
    # instatiate the class
    tees = TEES()
    tees.data_preprocess("custom-data.xml")
'''
    print "Reading dataset"
    dataset_name = "nyt"
    encoder="pcnn"
    selector="ave"
    myModel.read_dataset(dataset_name)
    print "Training"
    myModel.train(encoder, selector, dataset_name, epoch=1)
    print "Predicting"
    predictions = myModel.predict(encoder, selector, dataset_name)  # generate predictions! output format will be same for everyone
    print "Evaluating"
    myModel.evaluate(encoder, selector, dataset_name)
'''
