import abc

class GraphCompletion(abc.ABC):

	@abc.abstractmethod
	def read_dataset(self, fileName, options={}):
		"""
		Reads a dataset in preparation for: train or test. Returns data in proper format for: train or test.

		Args:
			fileName: Name of file representing the dataset to read
			options: object to store any extra or implementation specific data

		Returns:
			Iterable data, optionally split into train, test, and possibly dev.
		"""
		pass


	@abc.abstractmethod
	def train(self, data, options={}):
		"""
		Trains a model on the given input data

		Args:
			data: iterable of arbitrary format
			options: object to store any extra or implementation specific data

		Returns:
			ret: None. Trained model stored internally to instance's state. 
		"""
		pass


	@abc.abstractmethod
	def predict(self, data, options={}):
		"""
		Predicts on the given input data (e.g. knowledge graph). Assumes model has been trained with train()

		Args:
			data: iterable of arbitrary format. represents the data instances and features you use to make predictions
				Note that prediction requires trained model. Precondition: instance already stores trained model 
				information.
			options: object to store any extra or implementation specific data

		Returns:
			predictions: [tuple,...], i.e. list of predicted tuples. 
				Each tuple likely will follow format: (subject_entity, relation, object_entity), but isn't required.
		"""
		pass

	@abc.abstractmethod
	def evaluate(self, benchmark_data, metrics={}, options={}):
		"""
		Calculates evaluation metrics on chosen benchmark dataset.
		Precondition: model has been trained and predictions were generated from predict()

		Args:
			benchmark_data: Iterable testing split of dataset to evaluate on
			metrics: Dictionary of function pointers for desired evaluation metrics (e.g. F1, MRR, etc.)
				- Note: This abstract base class does not enforce a metric because some metrics are more appropriate 
				for a given benchmark than others. At least one metric should be specified
				- example format:
					metrics = {
						"F1": f1_eval_function,
						"MRR": mrr_eval_function
					}
			options: object to store any extra or implementation specific data

		Returns:
			evaluations: dictionary of scores with respect to chosen metrics
				- e.g.
					evaluations = {
						"f1": 0.5,
						"MRR": 0.8
					}
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