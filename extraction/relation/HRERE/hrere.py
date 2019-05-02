from optparse import OptionParser
from task import Task
import logging
import sys
import os
from model_param_space import param_space_dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from relation_extraction import *
from preprocess import *
from train import *

class HRERE(RelationExtraction):

	def __init__(self):
		pass


	def read_dataset(self, input_file, *args, **kwargs):  
		"""
		Reads a dataset to be used for training
         
         Note: The child file of each member overrides this function to read dataset 
         according to their data format.
         
		Args:
			input_file: Filepath with list of files to be read
		Returns: 
            (optional):Data from file
		"""
		pass



	#def data_preprocess(self,input_data, *args, **kwargs):
	def preprocess(data_name):
	    dataset = data_utils.DataSet(config.DATASET[data_name])

	    df_train, df_valid, df_test = dataset.load_raw_data()
	    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
	    train_size = df_train.shape[0]
	    test_size = df_test.shape[0]

	    e2id = dataset.save_e2id(set(list(df_all.e1) + list(df_all.e2)))
	    r2id = dataset.save_r2id(set(list(df_all.r)))

	    df_all.e1 = df_all.e1.map(e2id)
	    df_all.e2 = df_all.e2.map(e2id)
	    df_all.r = df_all.r.map(r2id)

	    df_train = df_all[:train_size]
	    df_valid = df_all[train_size:-test_size]
	    df_test = df_all[-test_size:]
	    dataset.save_data(df_train, df_valid, df_test)

	def parse_args(parser):
	    parser.add_option("-d", "--data", type="string", dest="data_name", default="fb15k")

	    options, args = parser.parse_args()
	    return options, args

	def main(options):
	    data_name = options.data_name
	    preprocess(data_name)
	


	def tokenize(self, input_data ,ngram_size=None, *args, **kwargs):  
		"""
		Tokenizes dataset using Stanford Core NLP(Server/API)
		Args:
			input_data: str or [str] : data to tokenize
			ngram_size: mention the size of the token combinations, default to None
		Returns:
			tokenized version of data
		"""
		pass


	#def train(self, train_data, *args, **kwargs):  
	def train(model_name, data_name, params_dict, logger, eval_by_rel, if_save):
	    task = Task(model_name, data_name, 1, params_dict, logger, eval_by_rel)
	    task.refit(if_save)

	def parse_args(parser):
		    parser.add_option("-m", "--model", dest="model_name", type="string", default="best_Complex_tanh_fb15k")
		    parser.add_option("-d", "--data", dest="data_name", type="string", default="fb15k")
		    parser.add_option("-r", "--relation", dest="relation", action="store_true", default=False)
		    parser.add_option("-s", "--save", dest="save", action="store_true", default=False)

		    options, args = parser.parse_args()
		    return options, args 
	def main(options):
	    logger = logging.getLogger()
	    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
	    train(options.model_name, options.data_name,
	        params_dict=param_space_dict[options.model_name],
	        logger=logger, eval_by_rel=options.relation, if_save=options.save)
		
		


	def predict(self, test_data, entity_1 = None, entity_2= None,  trained_model = None, *args, **kwargs):   
		"""
		Predict on the trained model using test data
		Args:
              entity_1, entity_2: for some models, given an entity, give the relation most suitable 
			test_data: test the model and predict the result.
			trained_model: the trained model from the method - def train().
						  None if store trained model internally.
		Returns:
              probablities: which relation is more probable given entity1, entity2 
                  or 
			relation: [tuple], list of tuples. (Eg - Entity 1, Relation, Entity 2) or in other format 
		"""
		pass

	
	def evaluate(self, input_data, trained_model = None, *args, **kwargs):
		"""
		Evaluates the result based on the benchmark dataset and the evauation metrics  [Precision,Recall,F1, or others...]
         Args:
             input_data: benchmark dataset/evaluation data
             trained_model: trained model or None if stored internally 
		Returns:
			performance metrics: tuple with (p,r,f1) or similar...
		"""
		pass
	

	def save_model(self, file):
		"""
		:param file: Where to save the model - Optional function
		:return:
		"""
		pass


	def load_model(self, file):
		"""
		:param file: From where to load the model - Optional function
		:return:
		"""
		pass


	if __name__ == "__main__":
	    parser = OptionParser()
	    options, args = parse_args(parser)
	    preprocess(data_name = options.data_name)
	    main(options)	
