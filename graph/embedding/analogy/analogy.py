from graph.embedding.graph_embedding import GraphEmbedding
from datetime import datetime
import logging
import numpy as np
import os


np.random.seed(46)

DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H%M')))


class ANALOGY(GraphEmbedding):

    def __init__(self, logger_path=None):

        if logger_path is None:
            logger_path = DEFAULT_LOG_DIR
        if not os.path.exists(logger_path):
            os.mkdir(logger_path)
        # if not os.path.exists(args.log):
        #    os.mkdir(args.log)
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO)
        log_path = os.path.join(logger_path, 'log')
        file_handler = logging.FileHandler(log_path)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(" Initializing ANALOGY ... ")

    def read_dataset(self, file_names, *args, **kwargs):  #<--- implemented PER class
        """ Reads datasets and convert them to proper format for train or test.
            Returns data in proper format for train, validation and test.

        Args:
            file_names: list-like. List of files representing the dataset to read. Each element is str, representing
            filename [possibly with filepath]
            options: object to store any extra or implementation specific data
        
        Returns:
            data: data in proper [arbitrary] format for train, validation and test.
        Raises:
            None
        """
        print("read data set")

        # <--- implemented PER class
    def learn_embeddings(self, data, *args, **kwargs):
        """
        Learns embeddings with data, build model and train the model

        Args:
            data: iterable of arbitrary format. represents the data instances and features and labels you need to train your model.
            Note: formal subject to OPEN ITEM mentioned in read_dataset!
            options: object to store any extra or implementation specific data
        Returns:
            ret: None. Trained model stored internally to class instance state.

        Raises:
            None
        """
        print("learn_embeddings")

    # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
    def evaluate(self, data, *args, **kwargs):
        """
        Predicts the embeddings with test data and calculates evaluation metrics on chosen benchmark dataset

        Args:
            data: data used to test the model, may need further process
            options: object to store any extra or implementation specific data

        Returns:
            metrics: cosine similarity, MMR or Hits

        Raises:
            None
        """

        results = {}
        return results
        print("evaluate")

    def save_model(self, file):
        """ saves model to file
        :param file: Where to save the model - Optional function
        :return:
        """
        print("save model")

    def load_model(self, file):
        """ loads model from file
        :param file: From where to load the model - Optional function
        :return:
        """
        print("load model")

"""
# Sample workflow:

inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

myModel = myClass(ditk.Graph_Embedding)  # instatiate the class

data = myModel.read_dataset(inputFiles)  # read in a dataset for training

myModel.learn_embeddings(data)  # builds and trains the model and stores model state in object properties or similar

results = myModel.evaluate(data)  # calculate evaluation results

print(results)

"""

