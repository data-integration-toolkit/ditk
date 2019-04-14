import abc
import csv
import os
import numpy as np
from random import sample

class Imputation(abc.ABC):
    # Any shared data structures or methods should be defined as part of the parent class.

    # A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.

    # The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group.

    @classmethod
    @abc.abstractmethod
    def preprocess(self, inputData, *args, **kwargs):
        """
		-Read's a dataset (complete dataset without missing values) and introduces missingness in the dataset. May also perform one or more of the following - 
		Scaling, masking, converting categorical data into one hot representation etc.
		
        Uses argparse() to get dataset name, output filename, data masking ratio, seed, prediction column
        and introduces missing values accordingly
		
		:param inputData: 
			FilePath to the (complete) dataset
		
        :param args:
            args.train_data: name of the processed file split for training
            args.test_data: name of the processed file split for testing
            args.valid_data: name of the processed file split for testing
            args.mask_ratio: percent of data to be randomly masked
            args.seed: seed of the randomizer
            args.predict_col: column used for prediction
        :param kwargs:
        :return:
            dataframe or numpy array or other data format
        """

        # creating path

        if not os.path.exist('checkpoint'):
            os.mkdir('checkpoint')

        if not os.path.exist('result'):
            os.mkdir('result')

        # loading csv data

        with open(inputData, 'r') as f:
            rows = csv.reader(f, delimiter=',', quotechar='|')
            data = [x for x in rows]
            data = np.asarray(data[1:], dtype='float')

        if args[0].split:
            test_index = sample(range(len(data)), int((1.0 - float(args[0].split))*len(data)))
            test_index.sort()
            training_index = [x for x in range(len(data)) if x not in set(test_index)]
            test_data = data[test_index]
            training_data = data[training_index]

            if not args[0].ims:
                return [(training_data, 'train', None), (test_data, 'test', None)]
            else:
                # handle introducing missing data here for both train and test
                pass

        # check if we introduce missing value here
        if not args[0].ims:
            return data, None
        else:
            # handing introducing missing data here, introduce missing data only for train
            pass

        # TODO change None to the mask
        return data, None

    @classmethod
    @abc.abstractmethod
    def train(self, train_data, *args, **kwargs):
        """
        Prepares the train_data and returns the trained model

        :param train_data: object returned by preprocess function
        :param args:
			args.validation_data: object returned by preprocess function
        :param kwargs:
        :return:
            trained_data: trained model
			(optional: test data splitted by the function)
        """
        pass

    @classmethod
    @abc.abstractmethod
    def test(self, trained_model, test_data, *args, **kwargs):
        """
        Picked up the trained model, load the test data, calculates performance for tuning

        :param trained_model: object returned by train function
        :param test_data: filepath to the test data or object representing the test data returned by train function
        :param args:
        :param kwargs:
        :return:
            imputed_data: imputed data in numpy or dataframe format
			(optional : performance_metric- rmse calculated)
        """
        pass


    """ 
    @classmethod
    @abc.abstractmethod
  
	def predict(self, trained_model, test_data, prediction=None, *args, **kwargs):
        
        Run prediction on determined column, if model already loaded from test, simply return the model and test data
        along with prediction columns

        :param trained_model: name of the trained model
        :param test_data: name of the test data
        :param prediction: column name to run prediction on
        :param args:
        :param kwargs:
        :return:
            model: the trained_model
            test_data: test data in numpy form
            prediction: col data to check against prediction
        
        pass
	"""


    @classmethod
    @abc.abstractmethod


    def impute(self, trained_model, input, *args, **kwargs):


        """
        Loads the trained_model and gives the imputed table

        :param trained_model: trained model returned by train function
        :param input: input table which needs to be imputed
        :param args:
        :param kwargs:
        :return:
            imputed_data: imputed table
        """
    pass


    @classmethod
    @abc.abstractmethod
    def evaluate(self, trained_model, input, *args, **kwargs):
        """
        Loads the trained_model and calculates the performance on the input through rmse.

        :param trained_model: trained model returned by train function
        :param input: input table on which model needs to be evaluated
        :param args:
        :param kwargs:
        :return:
            performance_metric: rmse
        """
        with open(input, 'r') as f:
            rows = csv.reader(f, delimiter=',', quotechar='|')
            data = [x for x in rows]
            data = np.asarray(data[1:], dtype='float')

        # check if we introduce missing value here
        if not args[0].ims:
            return None, data

        return None, data

    @abc.abstractmethod
    def save_model(self, file):
        """

        :param file: Where to save the model - Optional function
        :return:
        """
        pass

    @abc.abstractmethod
    def load_model(self, file):
        """

        :param file: From where to load the model - Optional function
        :return:
        """
        pass
