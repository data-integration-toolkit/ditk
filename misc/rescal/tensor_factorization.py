import abc

class TensorFactorization(abc.ABC):
	def __init__(self):
		pass
	
	@classmethod
	@abc.abstractmethod
	def read_dataset(self, path_tensor, key_tensor):
		'''
			Args:
				path_tensor: path to the tensor of size n*n*m in matlab (.mat) format
				key_tensor: name of the key to the tensor
		'''
		pass

	@classmethod
	@abc.abstractmethod
	def factorize(self, rank):
		'''
			Tensor factorization
			Args:
				rank: rank of the factorization

			Returns:
				matrix A of size n*k 
				a list of matrices R of length m, each element of the list is a matrix of size k*k
		'''
		pass

	@classmethod
	@abc.abstractmethod
	def save_model(self, dir):
		pass

	@classmethod
	@abc.abstractmethod
	def load_model(self, dir):
		pass

	@classmethod
	@abc.abstractmethod
	def evaluate(self):
		'''
			10-fold cross validation

			Returns:
				mean and standard deviation (SD) of Precision Recall Area Under Curve (PR AUC) for both training and test data
		'''
		pass