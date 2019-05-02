# -*- coding: utf-8 -*-

import numpy as np 

import theano
import theano.tensor as T

from utils import *

class Rec_OutputLayer(object): 
	"""
	Classic Output Layer for classification. We use a softmax to compute the conditional probabilities of tags given words, 
	and use them for computing the log-likelihood of the actual tag sequence and with an argmax for decoding.
	Inputs:
	input - theano tensor containing the hidden representations for the whole sentence
	n_in - dimension of the hidden representation
	n_out - dimension of tag vocabulary
	init - to choose from 'Initialization functions', will initiate the weight matrices (bias a are initiated with uniform distribution)
	""" 
	#init = init_zero
	def __init__(self, input, n_in, n_out, init):
	#print "nin", n_in
	#print "n_out", n_out
		self.W = globals()[init]((n_in, n_out), 'W')
		self.b = globals()[init]((n_out,), 'b')

		"""
		Again, no reccurence is needed, the scan is here for convenience when dealing with sequences of varying sizes.
		"""
		def rec_softmax(h_t):
			s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
			return s_t

		s, _ = theano.scan( fn=rec_softmax, \
			sequences = input, outputs_info = None, \
			)

		self.p_y_given_x = s[:,0,:]
		self.params = [(self.W, self.W), (self.b, self.b)]
		
	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


class Structured_OutputLayer(object):
	"""
	Output Layer used for structured prediction. We compute the score of the tags for each word, and use a weight matrix (W_tags) for the scores of transition from a tag to another in the viterbi procedure. (Score here is analog to log-probabilities : it is easier to use, ans safer, than probabilities)
	Inputs:
	input - theano tensor containing the hidden representations for the whole sentence
	n_in - dimension of the hidden representation
	n_out - dimension of tag vocabulary
	init - to choose from 'Initialization functions', will initiate the weight matrices (bias a are initiated with uniform distribution)
	"""
	def __init__(self, input, n_in, n_out, init):
		self.n_out = n_out
		self.W = globals()[init]((n_in, n_out), 'W')
		self.W_tags = globals()['init_uniform']((n_out, n_out), 'W_tags')
		self.b = globals()[init]((n_out,), 'b')
		self.t_0 = globals()['init_uniform']((n_out,), 't_0') 
	

		"""
		Simple procedure computing the scores of each tag for each word in the sentence - no recursion is needed, I use scan since the size of the sentence is variable.
		Helps to compute what we can compare to 'emission probabilities' in viterbi
		"""
		def rec_scores(x_t):
			s_t = (T.dot(x_t, self.W) + self.b)
			return s_t

		s, _ = theano.scan( fn=rec_scores, \
			sequences = input )

		self.p_y_given_x = s
		self.params = [(self.W, self.W), (self.b, self.b), (self.W_tags, self.W_tags)] 

	"""
	Wrapping the logsum function efficiently
	from : https://github.com/nouiz/lisa_emotiw/blob/master/emotiw/wardefar/crf_theano.py
	
	Compute log(sum(exp(x), axis=axis) in a numerically stable
	fashion.
	Parameters
	----------
	x : tensor_like
		A Theano tensor (any dimension will do).
	axis : int or symbolic integer scalar, or None
		Axis over which to perform the summation. `None`, the
		default, performs over all axes.
	Returns
	-------
	result : ndarray or scalar
		The result of the log(sum(exp(...))) operation.
	"""	
	def theano_logsumexp(self, x, axis=None):
		xmax = x.max(axis=axis, keepdims=True)
		xmax_ = x.max(axis=axis)
		return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

	"""
	Function computing the score of one tag sequence (=path) given the tag scores and the transition scores
	"""	
	def score_tag_path(self, y):

		def rec_obj(s_t, t_n1, t_n):
			acc = self.W_tags[t_n1,t_n] + s_t[t_n]
			return acc
	
		objs, _ = theano.scan( fn = rec_obj, \
			sequences = [ self.p_y_given_x[1:], dict(input= y, taps = [-1,0]) ],
			)
		return self.p_y_given_x[0,y[0]] + T.sum(objs)
	
	"""
	Reccurent functions that will be part of the forward and viterbi procedures. 
	"""	    
	def rec_forward(self, obs, d_t1, trans):
		d_t1 = d_t1.dimshuffle(0, 'x')
		out =  obs + self.theano_logsumexp(d_t1 + trans, axis = 0)
		return out

	def rec_viterbi(self, obs, d_t1, trans):
		d_t1 = d_t1.dimshuffle(0,'x')
		out_1 =  obs + (d_t1 + trans).max(axis = 0)
		out_2 = (d_t1 + trans).argmax(axis = 0)
		return out_1, out_2

	"""
	Computes the log_likelihood of the tag sequence by computing its score and using the forward procedure normalize it by computing the scores of all sequences.
	"""
	def negative_log_likelihood(self, y):
		logsum, _ = theano.scan( fn=self.rec_forward, \
			sequences = self.p_y_given_x[1:], outputs_info = self.p_y_given_x[0], non_sequences = self.W_tags\
			)
		return  - self.score_tag_path(y) +  self.theano_logsumexp(logsum[-1])

	"""
	In the decoding case, uses the viterbi procedure to find the highest-scoring tag sequence for the sentence. Only ouputs the last max scores and the argmax matrix that will be used
	outside of the network to get the best sequence.
	"""   
	def decode_forward(self):
		[max, argmax], _ = theano.scan( fn=self.rec_viterbi, \
			sequences = self.p_y_given_x[1:], outputs_info = [self.p_y_given_x[0], None], non_sequences = self.W_tags\
			)
		return max, argmax
