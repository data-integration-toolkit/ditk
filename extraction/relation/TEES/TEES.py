import TEES
from TEES.Detectors.Preprocessor import Preprocessor
from TEES import train
from TEES import classify
import sys
import os

class TEES():

    def __init__(self):
        pass

    def read_dataset(self):  
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
        
    def data_preprocess(self, input_data, output_file_path=os.getcwd()+"/output.xml", *args, **kwargs):
        """
         (Optional): For members who do not need preprocessing. example: .pkl files 
         A common function for a set of data cleaning techniques such as lemmatization, count vectorizer and so forth.
        Args: 
            input_data: Raw data to tokenize
        Returns:
            Formatted data for further use.
        """
        steps = "LOAD,BLLIP_BIO,STANFORD_CONVERT,SPLIT_NAMES,FIND_HEADS,SAVE"
        parseName = "McCC"
        requireEntities = False
        preprocessor = Preprocessor(steps, parseName, requireEntities)
        preprocessor.setArgForAllSteps("debug", False)
	preprocessor.process(input_data, output_file_path, model=None, logPath="AUTO")

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


    def train(self, train_file, devel_file, output_model_folder):  
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
        train.train(output_model_folder,inputFiles={"devel":devel_file, "train":train_file, "test":train_file}, models={"devel":"model-devel", "test":"model-test"}, parse="McCC", doFullGrid=False, log="log.txt", deleteOutput=False, debug=False)
        return output_model_folder + "/model-test"

    def predict(self, input_file, model_path, output_result_path):   
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
        classify.classify(input_file, model_path, output_result_path, omitSteps="PREPROCESS")
        return output_result_path+"-pred.xml.gz"

    def evaluate(self, input_file, model_path, gold_input, output_result_path):
        """
        Evaluates the result based on the benchmark dataset and the evauation metrics  [Precision,Recall,F1, or others...]
         Args:
             input_data: benchmark dataset/evaluation data
             trained_model: trained model or None if stored internally 
        Returns:
            performance metrics: tuple with (p,r,f1) or similar...
        """
        classify.classify(input_file, model_path, output_result_path, omitSteps="PREPROCESS", goldInput=gold_input)
        
        precision = None
        recall = None 
        f1_score = None
        # Analyze log file and retrive evaluation metrices.
        logFile = output_result_path+"-log.txt"
        lines = open(logFile, 'r').readlines()

        start_index = 0
        for i in range(len(lines)-1, -1, -1):
            if lines[i].split('\t')[1].replace('\n', '') == "Events":
                start_index = i
                break

        for i in range(start_index, len(lines)):
            if "micro" in lines[i]:
                metrices = lines[i].replace("\n","").split("micro ")[1].split(" ")
                positive_cases = metrices[0].split(":")[1].split("/")[0]
                negative_cases = metrices[0].split(":")[1].split("/")[1]

                positives = metrices[1].split(":")[1].split("|")[0]
                negatives = metrices[1].split(":")[1].split("|")[1]
                true_positive = positives.split("/")[0]
                false_positive = positives.split("/")[1]
                true_negative = negatives.split("/")[0]
                false_negative = negatives.split("/")[1]

                precision = float(metrices[2].split(":")[1].split("/")[0])
                recall = float(metrices[2].split(":")[1].split("/")[1])
                f1_score = float(metrices[2].split(":")[1].split("/")[2])
                break
        return precision, recall, f1_score

if __name__ == '__main__':
    tees = TEES()
    tees.data_preprocess("custom-data.xml", output_file_path=os.getcwd()+"/output.xml")
    test_model_path = tees.train(os.getcwd()+"/output.xml", os.getcwd()+"/output.xml", os.getcwd()+"/output.xml", os.getcwd()+"/Training_output")
    #test_model_path = os.getcwd() + "/Training_output" + "/model-test" 
    result_package = tees.predict(os.getcwd()+"/output.xml", test_model_path, os.getcwd()+"/Classify-Output")
    precision, recall, f1_score = tees.evaluate(os.getcwd()+"/output.xml", test_model_path, os.getcwd()+"/output.xml", os.getcwd() + "/Classify-Output")
    print(precision, recall, f1_score)

