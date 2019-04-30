from relation_extraction_2 import RelationExtractionModel

class clinicalRE(RelationExtractionModel):
	def __init__(self):
		RelationExtractionModel.__init__(self)

	# my own function

	def read_dataset(self, input_file, *args, **kwargs):
		'''
		Read train data & test data.
		Args:
			input_file: Filepath with list of files to be read
			dataset_name: The name of the dataset, will be one of [NYT, SemEval, DDI, i2b2]
		Returns: 
            (optional):Data from file
		'''
		pass

	def data_preprocess(self, input_data, *args, **kwargs):
		'''
		Args: 
			input_data: Raw data
			dataset_name: The name of the dataset.
		Returns:
			Formatted data for further use.
		'''
		pass

	# I will not use this function
	def tokenize(self, input_data ,ngram_size=None, *args, **kwargs):  
		pass

	def train(self, train_data, *args, **kwargs): 
		'''
		Args:
			train_data: Formatted data to be trained.	
        Returns: 
			None
		''' 
		pass

	def predict(self, test_data, entity_1 = None, entity_2= None,  trained_model = None, *args, **kwargs):
		'''
		Args:
			test_data: Formatted test data to be predicted. 
        Returns: 
			predictions
		''' 		
		pass

	def evaluate(self, input_data, trained_model = None, *args, **kwargs):
		'''
		Args:
			predictions: output of the predict function
			groud_truths: groud truth table
        Returns:
            metrics: tuple with (p,r,f1). Each element is float.
       
		'''
 		pass