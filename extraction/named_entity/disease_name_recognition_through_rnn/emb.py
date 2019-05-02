# -*- coding: utf-8 -*-
import numpy as np 
import theano
import theano.tensor as T
from utils import *

"""
fname: file name of embedded word vectors. Each line in the file represents a vector. 
n_f : is dimention of word vectors
"""
def readWV(fname, n_f):
	fw = open(fname, 'r')
	wv = []
	i = 0
	with open(fname, 'r') as f:
		for line in f :
			vs = line.split()
			vect = map(float, vs)
			if(len(vect) == n_f):
				wv.append(vect)
	w = np.asarray(wv, dtype='float64')
	return w


class LookupLayer(object):
	"""
	Classic Lookup Layer taking as inputs list of words id present in sentence word_id , the length of word vocabulary 'l_vocab_w' and the size of word embeddings 'n_f'.   The Lookup matrix 'WE' is the parameter and its gradient is computed with respect to 'output' to avoid computing updates for the whole matrix.
	"""
	def __init__(self, word_id, l_vocab_w, n_f, fname):
	# wv = readWV(fname, n_f)  # KRC UNCOMMENT if have pretrained word vocab
	# print "word embedding matrix shape ---------", wv.shape  # KRC UNCOMMENT if have pretrained word vocab
	# self.WE = theano.shared(value=np.asarray(wv, dtype='float64'), name='WE', borrow=True)  # KRC UNCOMMENT if have pretrained word vocab
		self.WE = globals()['init_zeros']((l_vocab_w, n_f), 'WE')  # KRC COMMENT if have pretrained word vocab
		self.output = self.WE[word_id]
		self.params = [(self.WE, self.output)]
