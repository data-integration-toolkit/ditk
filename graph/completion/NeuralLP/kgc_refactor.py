#import graph_completion
import sys,os
import argparse
import time
import tensorflow as tf
import numpy as np
from model import Learner
from data import Data, DataPlus
from experiment import Experiment
#from main import Option
import pickle
from collections import defaultdict
import graph_completion

#getting the path to the project
project_location = os.path.dirname(__file__)
print project_location
sys.path.insert(0,project_location+"/")

#import sys
#sys.path.insert(0,"/Users/tushyagautam/Documents/USC/Information_Integration/Project/Neural-LP-master/src/")

config = tf.ConfigProto()
# saver = tf.train.Saver()

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
        print d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))

class KGC_NeuralLP(graph_completion.GraphCompletion):

	option = ""
	data = ""
	experiment = ""
	saver = ""
	learner=""
	truths_file=""
	train_stats_1 = ""
	test_stats_1 = ""
	valid_stats_1 = ""

	def __init__(self):
		pass
		#:param d: dictionary to retrieve data

	def read_dataset(self, fileName, option=None):
		d={'resplit': False, 'vocab_embed_size': 128, 'seed': 33, 'exps_dir': 'exps/', 'rnn_state_size': 128, 'adv_rank': False, 'type_check': False, 'exp_name': 'demo', 'no_preds': False, 'no_train': False, 'rand_break': False, 'datadir': fileName, 'num_layer': 1, 'from_model_ckpt': None, 'no_rules': False, 'no_extra_facts': False, 'gpu': '', 'dropout': 0.0, 'min_epoch': 5, 'top_k': 10, 'max_epoch': 10, 'learning_rate': 0.001, 'print_per_batch': 3, 'no_norm': False, 'batch_size': 64, 'get_phead': False, 'num_step': 3, 'get_vocab_embed': False, 'query_is_language': False, 'query_embed_size': 128, 'thr': 1e-20, 'domain_size': 128, 'no_link_percent': 0.0, 'accuracy': False, 'rule_thr': 0.01}

		self.option=Option(d)
		print(self.option.datadir,
		self.option.exps_dir, 
		self.option.exp_name) 

		if self.option.exp_name is None:
		    self.option.tag = time.strftime("%y-%m-%d-%H-%M")
	    	else:
	      	    self.option.tag = self.option.exp_name  
	    	if self.option.resplit:
	      	    assert not self.option.no_extra_facts
	    	if self.option.accuracy:
	            assert self.option.top_k == 1
    
	    	os.environ["CUDA_VISIBLE_DEVICES"] = self.option.gpu
	    	tf.logging.set_verbosity(tf.logging.ERROR)
	       
	    	if not self.option.query_is_language:
	            self.data = Data(self.option.datadir, self.option.seed, self.option.type_check, self.option.domain_size, self.option.no_extra_facts) #directs to data.py
	    	else:
	            self.data = DataPlus(self.option.datadir, self.option.seed)
	    	print("Data prepared.")

	    	return self.data

	        
	
		'''
		Reads a dataset (either wn-18 or FB15) to prepare it for training and testing. 
		The folder is read and sent to the train module to process and ready for training and testing

		Args:
				self: A class instance
				fileName: List of files containing the dataset to read (provided along with filepath)

		'''

	

	


	def train(self):
		global saver

		self.option.num_entity = self.data.num_entity
	        self.option.num_operator = self.data.num_operator
	        if not self.option.query_is_language:
	            self.option.num_query = self.data.num_query
	        else:
	            self.option.num_vocab = self.data.num_vocab 
	            self.option.num_word = self.data.num_word # the number of words in each query

	        self.option.this_expsdir = os.path.join(self.option.exps_dir, self.option.tag)
	        if not os.path.exists(self.option.this_expsdir):
	            os.makedirs(self.option.this_expsdir)
	        self.option.ckpt_dir = os.path.join(self.option.this_expsdir, "ckpt")
	        if not os.path.exists(self.option.ckpt_dir):
	            os.makedirs(self.option.ckpt_dir)
	        self.option.model_path = os.path.join(self.option.ckpt_dir, "model")

	        self.option.save()
	        print("Option saved.")

	        # learner = Learner(self.option)
	        self.learner= Learner(self.option)

	        print("Learner built.")

	        saver = tf.train.Saver(max_to_keep=self.option.max_epoch)
	        saver = tf.train.Saver()

	        self.saver= tf.train.Saver(max_to_keep=self.option.max_epoch)
	        self.saver = tf.train.Saver()

	        config = tf.ConfigProto()
	        config.gpu_options.allow_growth = False
	        config.log_device_placement = False
	        config.allow_soft_placement = True
	        with tf.Session(config=config) as sess:
	            tf.set_random_seed(self.option.seed)
	            sess.run(tf.global_variables_initializer())
	            print("Session initialized.")

	            if self.option.from_model_ckpt is not None:
	                saver.restore(sess, self.option.from_model_ckpt)
	                print("Checkpoint restored from model %s" % self.option.from_model_ckpt)

	            self.data.reset(self.option.batch_size)
	            #this is a data object

	            self.experiment = Experiment(sess, saver, self.option, self.learner, self.data)
	            print("Experiment created.")

	            if not self.option.no_train:
		            print("Start training...")
		            self.experiment.train()

		            self.train_stats_1 = self.experiment.train_stats
		            self.valid_stats_1 = self.experiment.valid_stats
		            self.test_stats_1 = self.experiment.test_stats

		    

		

		'''
			Trains the model on the data. 
			Each tuple is of the form (relation, head, tail) where head and tail represent the entities in the tuple

			Args:
					self: A class instance

			Returns:
				dumps statistics into 'results.pckl' file

		'''

	def predict(self):        

		global config


		with tf.Session(config=config) as sess:
		    tf.set_random_seed(self.option.seed)
	            sess.run(tf.global_variables_initializer())
	            print("Session initialized.")

	            self.experiment = Experiment(sess, self.saver, self.option, self.learner, self.data)
	            print("Experiment created.")

		    if not self.option.no_preds:
			print("Start getting test predictions...")
			self.experiment.get_predictions()

	   	    if not self.option.no_rules:
			print("Start getting rules...")
			self.experiment.get_rules()

		'''
			Tests the model and performs predictions

			Returns:
				Writes to a file predicted entity relation entity triples
		'''

	def get_truths(self, folderPath):
		one_file=os.path.join(folderPath, "test.txt")
		text1_ptr = open(one_file,"r")
		text1=text1_ptr.readlines()
		one_file=os.path.join(folderPath, "valid.txt")
		text2_ptr = open(one_file,"r")
		text2=text2_ptr.readlines()

		one_file=os.path.join(folderPath, "train.txt")
		text3_ptr = open(one_file,"r")
		text3=text3_ptr.readlines()

		one_file=os.path.join(folderPath, "facts.txt")
		text4_ptr = open(one_file,"r")
		text4=text4_ptr.readlines()

		text="".join(text1)+"".join(text2)+"".join(text3)+"".join(text4)
		all_file = os.path.join(folderPath, "all_data.txt")
		f=open(all_file,"w+")
		f.writelines(text)




		all_file = os.path.join(folderPath, "all_data.txt")

		facts = []
		with open(all_file, "r") as f:
			for line in f:
				l = line.strip().split("\t")
				assert(len(l) == 3)
				facts.append(l)
		num_fact = len(facts)
		print("Number of all facts %d" % num_fact)

		query_head = defaultdict(list)
		query_tail = defaultdict(list)
		for h, r, t in facts:
			query_head[(r, h)].append(t)
			query_tail[(r, t)].append(h)

		to_dump = {}
		to_dump["query_head"] = query_head
		to_dump["query_tail"] = query_tail
		self.truths_file = os.path.join(folderPath, "truths.pckl")
		pickle.dump(to_dump, open(self.truths_file, "w"))

		print("Gather truths done.")

		return facts

		'''
			Retrieves all the facts(all triples from facts, train, test and valid)
			Sends these to the evaluate method to evaluate
		'''

	def evaluate(self, data, metrics={"hits@10", "MR", "MRR"}, options=None):

	    d={'preds':project_location+'/exps/demo/test_predictions.txt', 'raw':False, 'top_k':10, 'truths':self.truths_file, 'v':False}
	    option1=Option(d)
	    start = time.time()

	    if not option1.raw:
	        truths = pickle.load(open(option1.truths, "r"))
	        query_heads, query_tails = truths.values()
	    
	    hits = 0
	    hits_by_q = defaultdict(list)
	    ranks = 0
	    ranks_by_q = defaultdict(list)
	    rranks = 0.
	    line_cnt = 0

	    lines = [l.strip().split(",") for l in open(option1.preds).readlines()]
	    line_cnt = len(lines)

	    #print query_heads

	    for l in lines:
	        assert(len(l) > 3)
	        q, h, t = l[0:3]
	        this_preds = l[3:]
	        assert(h == this_preds[-1])
	        hitted = 0.

	        if not option1.raw:
	            if q.startswith("inv_"):
	                q_ = q[len("inv_"):]
	                also_correct = query_heads[(q_, t)]
	            else:
	                also_correct = query_tails[(q, t)]
	            also_correct = set(also_correct)
	            assert(h in also_correct)
	            #this_preds_filtered = [j for j in this_preds[:-1] if not j in also_correct] + this_preds[-1:]
	            this_preds_filtered = set(this_preds[:-1]) - also_correct
	            this_preds_filtered.add(this_preds[-1])
	            if len(this_preds_filtered) <= option1.top_k:
	                hitted = 1.
	            rank = len(this_preds_filtered)
	        else:
	            if len(this_preds) <= option1.top_k:
	                hitted = 1.
	            rank = len(this_preds)

	        # if len(this_preds) <= option1.top_k:
	        # 	hitted = 1.
	        # rank = len(this_preds)
	        
	        hits += hitted
	        ranks += rank
	        rranks += 1. / rank
	        hits_by_q[q].append(hitted)
	        ranks_by_q[q].append(rank)

	    print("Hits at %d is %0.4f" % (option1.top_k, hits / line_cnt))
	    print("Mean rank %0.2f" % (1. * ranks / line_cnt))
	    print("Mean Reciprocal Rank %0.4f" % (1. * rranks / line_cnt))

	    if option1.v:
	        hits_by_q_mean = sorted([[k, np.mean(v), len(v)] for k, v in hits_by_q.items()], key=lambda xs: xs[1], reverse=True)
	        for xs in hits_by_q_mean:
	          xs += [np.mean(ranks_by_q[xs[0]]), np.std(ranks_by_q[xs[0]])]
	          print(", ".join([str(x) for x in xs]))

	    print("Time %0.3f mins" % ((time.time() - start) / 60.))
	    print("="*36 + "Finish" + "="*36)

	    Hits1 = (option1.top_k, hits / line_cnt)
	    MR1 = (1. * ranks / line_cnt)
	    MRR1 = (1. * rranks / line_cnt)

	    return {'Hits@10':Hits1,'MR':MR1,'MRR':MRR1}

	    '''
			retrieves predicted triples from test_predictions.txt
			Performs evaluation metrics Hits@10, MR, and MRR on the set of triples
	    '''

	def save_model(self, save_Path):
		"""
		:param file: Where to save the model - Optional function
		:return:
		"""

		pickle.dump([self.train_stats_1, self.valid_stats_1, self.test_stats_1],
                    open(save_Path+'/results_saved.pckl', "w"))

		pass



	def load_model(self, save_Path):
		"""
		:param file: From where to load the model - Optional function
		:return:
		"""

		file_obj = open(save_Path+'/results_saved.pckl', 'rb')
		random = pickle.load(file_obj)

# def main():

# 	obj=KGC_NeuralLP()
# 	obj.read_dataset('/Users/tushyagautam/Documents/USC/Information_Integration/Project/Neural-LP-master/src/datasets/kinship')
# 	obj.train()
# 	obj.predict()
# 	obj.get_truths('/Users/tushyagautam/Documents/USC/Information_Integration/Project/Neural-LP-master/src/datasets/kinship')
# 	obj.evaluate(obj.truths_file)

# if __name__ == "__main__":
# 	main()
