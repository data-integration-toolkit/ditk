import abc

class Blocking(abc.ABC):

	# Any shared data strcutures or methods should be defined as part of the parent class.
	
	# A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
	
	# The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group. 
	
	def __init__(self, blocking_module, *args, **kargs):
		"""
		Initialise object
		Args:
			blocking_module: String that mentions the blocking module being used.
                        **kargs: Dictionary containing parameters needed in blocking pipeline. 
		"""
		self.blocking_module = blocking_module

	@abc.abstractmethod
	def read_dataset(self, filepath_list, *args, **kwargs):
		"""
		Accepts list of URI's and returns a List of Pandas DataFrames.
		Args:
			filepath_list: List of string path to the dataset.
			*args: Additional arguments can be mentioned.
                        **kargs: Parameters for reading datasets.

		Returns: List of Pandas DataFrames
		"""
		pass

	@abc.abstractmethod
	def train(self, dataframe, *args, **kwargs):
		"""
		Accepts Pandas Dataframe to train the model on and returns the trained model
		Args:
			dataframe: Pandas dataframe with training vectors under 'vectors' column. Row id's 
                            will be used as node id's in the tree built by the related pairs algorithms. 
			*args: Empty
                        **kargs: Dictionary containing requisite parameters for training/building the 
                            algorithm specified in their original papers/githubs. 

		Returns: Trained model
		"""
		pass

	@abc.abstractmethod
	def predict(self, model, dataframe_list, *args, **kwargs):
		"""
		Given a trained model and Pandas Dataframe of datasets, predict() returns a list of pairs of
		    related ids as a list of tuples.
		Args:
			model: tensorflow model used to predict.
			dataframe_list: List of Pandas Dataframes.
			*args: Additional arguments.
                        **kargs: Parameters requisite for prediciting related ids as defined by the original papers/githubs. 

		Returns: List of pairs of elements related in the dataset. Lists of Lists of tuples [[(id_0,0.99),(id_3,0.95)],...,[(id_5,0.97),(id_1,0.83)]].
		"""
		pass

	@abc.abstractmethod
	def evaluate(self, groundtruth, dataframe_list, *args, **kwargs):
		"""
		Given the ground truth and list of dataframes to predict the related ids for, evaluate() returns the Precision,
		    Recall and Reduction Ratio metrics.
		Args:
			groundtruth: String path to or Dataframe of the ground truth data.
			dataframe_list: List of Pandas Dataframes with the data to predict the related id's for. 
			*args: If more than 1 dataset, string dataset path can be mentioned as additional arguments.
                        **kargs: Parameters requisite for prediciting related ids as defined by the original papers/githubs.

		Returns: Precision, Recall, Reduction_Ratio
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