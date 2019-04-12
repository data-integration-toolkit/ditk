import abc

class GraphEmbedding(abc.ABC):

	@abc.abstractmethod
	def read_dataset(self, file_names, *args, **kwargs):  #<--- implemented PER class
		"""
		Reads datasets and convert them to proper format for train or test. Returns data in proper format for train, validation and test.

		Args:
			file_names: list-like. List of files representing the dataset to read. Each element is str, representing
				filename [possibly with filepath]
                        options: object to store any extra or implementation specific data

		Returns:
			data: data in proper [arbitrary] format for train, validation and test.
		Raises:
			None
		"""

		pass


	@abc.abstractmethod
	def learn_embeddings(self, data, *args, **kwargs):  #<--- implemented PER class
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

		pass



	@abc.abstractmethod
	def evaluate(self, data, *args, **kwargs):  #<--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
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

		# return results
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

"""
# Sample workflow:

inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

myModel = myClass(ditk.Graph_Embedding)  # instatiate the class

data = myModel.read_dataset(inputFiles)  # read in a dataset for training

myModel.learn_embeddings(data)  # builds and trains the model and stores model state in object properties or similar

results = myModel.evaluate(data)  # calculate evaluation results

print(results)

"""
