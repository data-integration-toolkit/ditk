import unittest
import pandas as pd
import matplotlib.pyplot as plt, uuid, sys, os, time, argparse
# from graph.completion.graph_completion import graph_completion
from main import GraphEmbeddingChild
import tensorflow as tf


class TestGraphEmbeddingMethods(unittest.TestCase):

	def setUp(self):
		self.input_file = 'data/yago/large/testcase.txt'
		parser = argparse.ArgumentParser(description='HyTE')
		parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wiki_data'], help ='dataset to choose')
		parser.add_argument('-version',dest = 'version', default = 'large', choices = ['large','small'], help = 'data version to choose')
		parser.add_argument('-test_freq',    dest="test_freq",  default = 25,       type=int,   help='Batch size')
		parser.add_argument('-neg_sample',   dest="M",      default = 5,    type=int,   help='Batch size')
		parser.add_argument('-gpu',      dest="gpu",        default='1',            help='GPU to use')
		parser.add_argument('-model',    dest="model",      default='hyte_2_08_04_2019_14:42:12',help='Name of the run')
		parser.add_argument('-drop',     dest="dropout",    default=1.0,    type=float, help='Dropout for full connected layer')
		parser.add_argument('-rdrop',    dest="rec_dropout",    default=1.0,    type=float, help='Recurrent dropout for LSTM')
		parser.add_argument('-lr',   dest="lr",         default=0.0001,  type=float,    help='Learning rate')
		parser.add_argument('-lam_1',    dest="lambda_1",       default=0.5,  type=float,   help='transE weight')
		parser.add_argument('-lam_2',    dest="lambda_2",       default=0.25,  type=float,  help='entitty loss weight')
		parser.add_argument('-margin',   dest="margin",     default=10,     type=float,     help='margin')
		parser.add_argument('-batch',    dest="batch_size",     default= 50000,     type=int,   help='Batch size')
		parser.add_argument('-epoch',    dest="max_epochs",     default= 25,    type=int,   help='Max epochs')
		parser.add_argument('-l2',   dest="l2",         default=0.0,    type=float,     help='L2 regularization')
		parser.add_argument('-seed',     dest="seed",       default=1234,   type=int,   help='Seed for randomization')
		parser.add_argument('-inp_dim',  dest="inp_dim",    default = 128,      type=int,   help='Hidden state dimension of Bi-LSTM')
		parser.add_argument('-L1_flag',  dest="L1_flag",    action='store_false',           help='Hidden state dimension of FC layer')
		parser.add_argument('-onlytransE', dest="onlytransE",   action='store_true',        help='Evaluate model on only transE loss')
		parser.add_argument('-restore',  dest="restore",    action='store_false',       help='Restore from the previous best saved model')
		parser.add_argument('-res_epoch',        dest="restore_epoch",  default=75,   type =int,        help='Restore from the previous best saved model')
		args = parser.parse_args()
		args.dataset = 'data/'+ args.data_type +'/'+ args.version+'/train.txt'
		args.entity2id = 'data/'+ args.data_type +'/'+ args.version+'/entity2id.txt'
		args.relation2id = 'data/'+ args.data_type +'/'+ args.version+'/relation2id.txt'
		args.test_data  =  self.input_file # 'data/'+ args.data_type +'/'+ args.version+'/testcase.txt'
		args.triple2id  =   'data/'+ args.data_type +'/'+ args.version+'/triple2id.txt'
			# args = Namespace(L1_flag=True, M=5, batch_size=50000, data_type='yago', dataset='data/yago/large/train.txt', dropout=1.0, entity2id='data/yago/large/entity2id.txt', gpu='1', inp_dim=128, l2=0.0, lambda_1=0.5, lambda_2=0.25, lr=0.0001, margin=10, max_epochs=25, model='hyte_2_08_04_2019_14:42:12', onlytransE=False, rec_dropout=1.0, relation2id='data/yago/large/relation2id.txt', restore=True, restore_epoch=75, seed=1234, test_data='data/yago/large/testcase.txt', test_freq=25, triple2id='data/yago/large/triple2id.txt', version='large')

		self.graph_embedding = GraphEmbeddingChild(args) # initializes your Graph Embedding class

	def test_read_dataset(self):
		train, validation, test = self.graph_embedding.read_dataset()
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(train, list) # assert non-empty list
		self.assertTrue(validation, list) # assert non-empty list
		self.assertTrue(test, list) # assert non-empty list

	def test_evaluate(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			self.graph_embedding.set_session(sess)
			evaluations = self.graph_embedding.evaluate(self.input_file)

		# Evaluations could be a dictionary or a sequence of metrics names

		self.assertIsInstance(evaluations, dict)
		self.assertIn("test_tail rank", evaluations)
		self.assertIn("test_head rank", evaluations)
		self.assertIn("test_tail HIT@10", evaluations)
		self.assertIn("test_head HIT@10", evaluations)

		# tail_rank, head_rank, tail_hit, head_hit = self.graph_embedding.evaluate(self.input_file)
		self.assertIsInstance(evaluations["test_tail rank"], float)
		self.assertIsInstance(evaluations["test_head rank"], float)
		self.assertIsInstance(evaluations["test_tail HIT@10"], float)
		self.assertIsInstance(evaluations["test_head HIT@10"], float)


if __name__ == '__main__':
    unittest.main()
