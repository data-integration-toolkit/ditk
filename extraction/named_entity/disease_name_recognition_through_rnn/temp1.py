# 					Disease mention recognition 

#This code is a modified version of the code provided by Matthieu Labeau in "Non-lexical neural architecture for fine-grained POS Tagging" https://github.com/MatthieuLabeau/NonlexNN

import sys

from model1 import *
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

from data_proc import *
from dataset import Dataset
from utils import *
from emb import *
from hidden import *
from output import *
from trainer import *

from six.moves import cPickle

sys.setrecursionlimit(5000)

# ---------------------Helper for saving model-------------------- #
import copy_reg
import types
import multiprocessing

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

# -------------------End Helper for saving model------------------ #

def model_train(N,M,n_f,n_hidden,model,viterbi,trainer,lr,batch_size,wl,id2word,tl,save_checkpoint_models=False):

	# N = 501#01 			# Number of iteration train 
	# M = 100#00 			# after every 10k sentence, check validation and testing datas
	# n_f = 100 			# dim of word embedding
	# n_hidden = 100 			# number of nodes in hidden layer
	# model = 'rnn' 			# model used for classification : 'ff' or 'rnn'
	# viterbi = True 			# structure output or not
	# trainer = 'AdagradTrainer' 	# Training method
	# lr = 0.05 			# learning rate
	# batch_size = 1 			# batch_size 
	activation = T.tanh		# activation function
	output_fname_dev = "results/rnn_dev_res_checkpoint_"		# output file for dev result
	output_fname_test = "results/rnn_test_res_checkpoint_"		# output file for test result

	# input file names
	path = "binary_data/"
	train_w = "train_words.txt"
	train_t = "train_tags.txt"
	dev_w = "dev_words.txt"
	dev_t = "dev_tags.txt"
	test_w = "test_words.txt"
	test_t = "test_tags.txt"
	tag = "tag.txt"

	# input file names, spans
	# train_spans = 'train_spans.txt'
	# dev_spans = 'dev_spans.txt'
	# test_spans = 'test_spans.txt'

	# fvocab = "word2vec/vocab.txt"		# word list
	fname = "word2vec/cbow_100d.txt"	# its corresponding vector we have assigne random vector for OoV word [NOT YET IMPLEMENTED. requires changes to emb.py]

	# Creating vocabularies 
	# wl = create_wl(fvocab)
	print "word len", len(wl)

	# id2word = invert_dict(wl)

	# tl = {"O": 0, "B-Dis": 1, "I-Dis":2}

	trainset = Dataset(path+train_w, path+train_t, wl, tl, batch_size)
	devset = Dataset(path+dev_w, path+dev_t, wl, tl, batch_size)
	testset = Dataset(path+test_w, path+test_t, wl, tl, batch_size)

	print "trainset", trainset.tot
	print "devset", devset.tot
	print "testset", testset.tot

	sampler = trainset.sampler()
	dev_sampler = devset.sampler()
	test_sampler = testset.sampler()

	l_vocab_w = len(wl)
	l_vocab_out=len(tl) 

	model = SLLModel(l_vocab_w = l_vocab_w, l_vocab_out = l_vocab_out, n_f = n_f, n_hidden = n_hidden, lr = lr, trainer = trainer, activation=T.tanh, model = model, viterbi = viterbi, fname = fname, wl = wl)

	#train_res = []
	j=1
	for i in range(N):
		#print "N iter number: ", i
		if(i == 0):
			continue
		inputs, tags, = sampler.next()
		res = model.train_perplexity(inputs[0],tags[0])
		if(i%500 == 0):
			print res
		if(i%M == 0):
			print "res", res
			f = open(output_fname_dev +str(j),"w+")
			dev_pred = []
			dev_true = []
					#DEV DATA
			print "starting dev evals on N iter number: ", i
			for m in range(devset.tot):
	 			dev_inputs, dev_tags  = dev_sampler.next()		#dev data
				res = model.eval_perplexity(dev_inputs[0], dev_tags[0])
				pred = model.predict(dev_inputs[0])

				viterbi_max, viterbi_argmax =  model.output_decode(dev_inputs[0]) 	
				first_ind = np.argmax(viterbi_max[-1])
				viterbi_pred =  backtrack(first_ind, viterbi_argmax)
				vi_pre = np.array(viterbi_pred)
				dev_true = list(dev_tags[0])
				for k,l,n in zip(vi_pre, dev_true, dev_inputs[0]):
					dev_true.extend(str(l))
					dev_pred.extend(str(k))
					f.write(id2word[n])
					f.write(" ")
					if(l == 0):
						f.write("O")
					if(l == 1):
						f.write("B-Dis")
					if(l == 2):
						f.write("I-Dis")	
					f.write(" ")
					if(k == 0):
						f.write("O")
					if(k == 1):
						f.write("B-Dis")
					if(k == 2):
						f.write("I-Dis")
					f.write('\n')
				f.write('\n')
			f.close()

			g = open(output_fname_test +str(j),"w+")
			j += 1
					#TEST DATA
			print "starting test evals on N iter number: ", i
			for m in range(testset.tot):
				test_inputs, test_tags = test_sampler.next()		#dev data
				res = model.eval_perplexity(test_inputs[0], test_tags[0])
				pred = model.predict(test_inputs[0])
				viterbi_max, viterbi_argmax =  model.output_decode(test_inputs[0]) 	
				first_ind = np.argmax(viterbi_max[-1])
				viterbi_pred =  backtrack(first_ind, viterbi_argmax)
				vi_pre = np.array(viterbi_pred)
				test_true = list(test_tags[0])
				for k,l,n in zip(vi_pre, test_true, test_inputs[0]):
					g.write(id2word[n])
					# print n
					g.write(" ")
					if(l == 0):
						g.write("O")
					if(l == 1):
						g.write("B-Dis")
					if(l == 2):
						g.write("I-Dis")	
					g.write(" ")
					if(k == 0):
						g.write("O")
					if(k == 1):
						g.write("B-Dis")
					if(k == 2):
						g.write("I-Dis")
					g.write('\n')
				g.write('\n')			
			g.close()
			if save_checkpoint_models:
				print('pickling model for checkpoint number %s'%j)
				checkpointFileName = 'trained_model_ckpt_%s.save'%j
				with open(checkpointFileName,'wb') as sm:
					cPickle.dump(model, sm, protocol=cPickle.HIGHEST_PROTOCOL)
				print('pickled model for checkpoint number %s into file: %s'%(j,checkpointFileName))

	return model

