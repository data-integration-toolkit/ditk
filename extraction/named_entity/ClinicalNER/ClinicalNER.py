from ner import NER

class clinicalNER(NER):
	def convert_ground_truth(self, data, *args, **kwargs):
		pass

	def read_dataset(self, fileNames, *args, **kwargs):
		'''
		Args:
            fileNames: list-like. List of files representing the dataset to read. Each element is str, representing
                filename [possibly with filepath]
        Returns:
            data: data in arbitrary format for train or test.
        '''
		pass

	def convert_data_format(self, data, dataset_name, *args, **kwargs):
		'''
		Args: 
			data: Raw data
			dataset_name: The name of the dataset.
		Returns:
			Formatted data for further use.
		'''
		pass

	def train(self, data, *args, **kwargs):
		'''
		Args:
			data: Formatted data to be trained.
		Returns:
			None
		'''
		pass

	def predict(self, data, *args, **kwargs):
		'''
        Args:
            data: Formatted test data to be predicted.

        Returns:
            predictions: [tuple,...], i.e. list of tuples.
        '''
		pass

	def evaluate(self, predictions, groudTruths, *args, **kwargs):
		'''
		Args:
            predictions: [tuple,...], list of tuples [same format as output from predict]
            groundTruths: [tuple,...], list of tuples representing ground truth.

        Returns:
            metrics: tuple with (p,r,f1). Each element is float.
		'''
		pass
