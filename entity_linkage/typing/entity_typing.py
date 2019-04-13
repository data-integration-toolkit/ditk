'''
Group5. Entity Typing 
'''

#GitHub Link - https://github.com/easy1one/Entity_Type_api
#Parent class - https://github.com/easy1one/Entity_Type_api/blob/master/entity_typing.py

import abc

class entity_typing(abc.ABC):

	@abc.abstractmethod
	def read_dataset(self, file_names, options={}):
		''' 
		input: List of Strings file_names OR file_paths to each file needed for the module
		output: a tuple of lists containing the dataset for train_data(tasks) and test_data(tasks) each as lists and other data as needed per project

		'''
		pass

	@abc.abstractmethod
	def train(self, train_data, options={}) :
		''' 
		input: train_data a list of the training data returned from the read_dataset method required for the module
		output: None or optionally model details if it's not returned then it will save the model into a file OR store within the system module
		'''
		pass

	@abc.abstractmethod
	def predict(self, test_data, model_details=None, options={}) :
		''' 
		input: list containing the test input, optionally include the model_details if returned from train
		output: iterable containing the predicted output
		'''
		pass


	@abc.abstractmethod
	def evaluate(self, test_data, prediction_data=None,  options={}) :
		''' 
		input: list containing the test input, and optionally the prediction_data which if not present will cause the predict method to be run.
		output: list contatining all output data including (f1 score, and MRR if present); set default value if the code is not applicable
		'''

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
