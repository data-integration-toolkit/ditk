import abc
import numpy as np
from scipy.stats import pearsonr


class TextSemanticSimilarity(abc.ABC):

	def __init__(self):
		pass

	@abc.abstractmethod
	def read_dataset(self, file_name, *args, **kwargs):
		"""
		Reads a dataset that is a CSV/Excel File.

		Args:
			file_name : With it's absolute path

		Returns:
			training_data_list : List of Lists that containes 2 sentences and it's similarity score 
			Note :
				Format of the output : [[S1,S2,Sim_score],[T1,T2,Sim_score]....]

		Raises:
			None
		"""
		# parse files to obtain the output
		# return training_data_list
		pass

	@abc.abstractmethod
	def train(self, data_X, data_Y, *args, **kwargs):  # <--- implemented PER class

		# some individuals don't need training so when the method is extended, it can be passed

		pass

	@abc.abstractmethod
	def predict(self, data_X, data_Y, *args, **kwargs):
		"""
		Predicts the similarity score on the given input data(2 sentences). Assumes model has been trained with train()

		Args:
			data_X: Sentence 1(Non Tokenized).
			data_Y: Sentence 2(Non Tokenized)

		Returns:
			prediction_score: Similarity Score ( Float ) 
				
		Raises:
			None
		"""
		pass

	@abc.abstractmethod
	def generate_embeddings(self, input_list, *args, **kwargs):
		"""
		Returns the embeddings that have been used to compute similarity ( Hash, Word2Vec, GlovE..)

		Args:
			input_list : List of Words

		Returns:
			embeddings_list : List of embeddings/hash of those words

		Raises:
			None
		"""
		pass

	@abc.abstractmethod
	def evaluate(self, actual_values, predicted_values, *args, **kwargs):
		"""
		Returns the correlation score(0-1) between the actual and predicted similarity scores

		Args:
			actual_values : List of actual similarity scores
			predicted_values : List of predicted similarity scores

		Returns:
			correlation_coefficient : Value between 0-1 to show the correlation between the values(actual and predicted)

		Raises:
			None
		"""

		x = np.array(actual_values)

		y = np.array(predicted_values)

		r, p = pearsonr(x, y)


		evaluation_score = r

		return evaluation_score


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

"""
# Sample workflow:

inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

myModel = myClass(DITKModel_NER)  # instatiate the class

train_X, train_Y = myModel.read_dataset(inputFiles,'train')  # read in a dataset for training
test_X, test_Y = myModel.read_dataset(inputFiles,'test')  # read in a dataset for testing

myModel.train(train_X,train_Y)  # trains the model and stores model state in object properties or similar

predictions = myModel.predict(test_X)  # generate predictions! output format will be same for everyone

P,R,F1 = myModel.evaluate(predictions, test_Y)  # calculate Precision, Recall, F1

print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))

"""
