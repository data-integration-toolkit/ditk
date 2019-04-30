import abc
class TextEmbedding(abc.ABC):
    
    # This class defines the common behavior that the sub-classes in the family of Text Embedding can implement/inherit
    # Any shared data strcutures or methods should be defined as part of the parent class.
    # A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
    # The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group.

    """
    BENCHMARKS:
        -----------------------------------------------------------------------------------------------------------------------------------------------
        |   DATASET             |         FORMAT                    |               EXAMPLE                      |    EVALUATION METRICS               |     |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | Cornell Movie Reviews | reviews and its sentiment         |"uncompromising french director robert      |   Precision, Recall, F1             |
        | Sentiment Analysis    |                                   | bresson's " lancelot of the lake...", pos  |                                     |     |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | CoNll2003: NER        | entity and its type               | ["LOC","Afghanistan"]                      |   Precision, Recall, F1             |     |
        |----------------------------------------------------------------------------------------------------------------------------------------------- 
        | CategoricalDataset    | data and its category             | ["Office Services Coordinator", 69222.18]  |   Mean Square Error                 |     |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | SemEval: Similarity   | sentences and its similarity score|['Dogs are fighting','Dogs are wrestling',4]|   Pearson Correlation Coefficient   |          | 
        |                       |                                   |                                            |                                     |                                   Coefficient                |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | SICK Dataset          | sentences and its similarity score|['Dogs are fighting','Dogs are wrestling',4]| Pearson Correlation Coefficient     |
        |                       |                                   |                                            |                                     |
        ------------------------------------------------------------------------------------------------------------------------------------------------
    """    

    def __init__(self):
        """
        Shared data members initialized in the constructor -- 
        sentences -- input for each model as required by each method
        benchmarks -- list of pre-defined benchmarks as strings
        is_benchmark -- boolean value to indicate if the dataset read is a benchmark or not
        metrics -- dictionary of evaluation metrics along with their computed values after testing
        """
        self.sentences = []
        self.benchmarks = ['conll2003', 'cornellMD', 'categorical', 'semEval','sick']
        self.is_benchmark = False
        self.metrics = {
            'conll2003': [
                'precision',
                'recall',
                'F1'
            ],
            'cornellMD': [
                'precision',
                'recall',
                'F1'
            ],
            'categorical': [
                'mse'
            ],
            'semEval': [
                'pearson_coeff'
            ]
             'semEval': [
                'pearson_coeff'
            ]
        }

    @classmethod
    @abc.abstractmethod
    def read_Dataset(self, name, fileName):
        """
        Task - reads the dataset "name" provided by the user in the "fileName" path. Can be a benchmark dataset or a list of tokens
        
        Input:
        name -- string -- Can be one of the following : ['conll2003', 'semEval', 'categorical','sick'] or a method specific dataset
        fileName -- string -- Directory Path to the dataset
        return:
        stores the dataset in method appropriate format in self.sentences
        """
        pass

    @classmethod
    @abc.abstractmethod
    def train(self, *argv, **kwargs):
        """
        Task - Train the embedding model using the pre-loaded dataset and method specific arguments
        
        Input: 
        filepath -- string -- path where the user wants the trained model to be stored
    
        Action - Saves the trained model in the user specified directory path
        """
        pass

    @classmethod
    @abc.abstractmethod
    def predict_embedding(self, input):
        """
        Task - Predicts and returns the best embedding for the token in the input list
        Input: 
        input -- list -- List of token(s)
        Action - Predicts the embedding vector for a string or an average vector for a sentence/corpus based on the pre-trained or custom-trained model
        return: 
        embedding -- vector -- a vector embedding for the token/ list of tokens
        """

    @classmethod
    @abc.abstractmethod
    def predict_similarity(self, input1, input2):
        """
        Task - Predicts the similarity between the embeddings of input1 and input2 using the 
        
        Input: 
        input1 -- list -- list of token(s) representing input1
        input2 -- list -- list of token(s) representing input2
        Action - Calculates the embeddings/average embeddings for the two inputes and computes the similarity between them
        return: 
        similarity score -- float -- word mover distance from gensim
        """ 
    @classmethod
    @abc.abstractmethod
    def evaluate(self, model, filename, evaluation_type):
        """
        Task - Evaluates a pre-trained model on the test dataset and returns the evaluation along with metrics as a dictionary
        Input:
        model -- string -- path to the pre-trained model
        filename -- string -- path to the evaluation/test dataset
        evaluation_type -- string -- specific to the type of evaluation being made. Can be ['ner', 'semeval', 'categorical']
        Action - Performs the user specified evaluation on the test dataset using the pretrained model
        
        return: 
        Dictionary of {evaluation_metric<string>:calculated_value<float>}
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