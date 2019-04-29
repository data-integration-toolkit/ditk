#Python 3.x

import abc 

class Imputation(abc.ABC):

    @abc.abstractmethod
    def preprocess(self, input_file, *args, **kwargs):
        """
	   Reads a dataset (complete dataset without missing values) and introduces missingness in the dataset.
        
        May also perform one or more of the following - 
	   Scaling, masking, converting categorical data into one hot representation etc.
		
        Uses argparse() to get dataset name, output filename, data masking ratio, seed, prediction column
        and introduces missing values accordingly
		
		
        :param input_file:
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
        pass

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
		   (optional: test data split)
        """
        pass

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
    Since the imputation group primarily aims to impute the table, we have not opted for
    a predict function for classification, however, if classification needs to be done
    in the future, this abstract function can be used to predict
    
    @abstractmethod
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

    @abc.abstractmethod
    def impute(self, trained_model, model_input, *args, **kwargs):
        """
     Loads the trained_model and gives the imputed table
    
    	:param trained_model: trained model returned by train function
    	:param model_input: input table which needs to be imputed
    	:param args:
    	:param kwargs:
    	:return:
    		imputed_data: imputed table
        """
        pass
    

    @abc.abstractmethod
    def evaluate(self, trained_model, model_input, *args, **kwargs):
        """
        Loads the trained_model and calculates the performance on the input through rmse.

        :param trained_model: trained model returned by train function
        :param model_input: input table on which model needs to be evaluated
        :param args:
        :param kwargs:
        :return:
            performance_metric: rmse
        """
        pass

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