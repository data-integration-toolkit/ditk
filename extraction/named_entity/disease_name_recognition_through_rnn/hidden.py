# -*- coding: utf-8 -*-

import numpy as np 

import theano
import theano.tensor as T

from utils import *

class FFLayer(object):
	"""
	Classic feedforward Layer.
	Inputs:
	input - theano tensor containing the sequence of embeddings for the whole sentence
	n_in - dimension of the embeddings
	winDim - size of the window taken as context each side of the word 
	n_out - dimension of the hidden representation
	activation - activation function used in the layer
	init - to choose from 'Initialization functions', will initiate the weight matrix (bias and 'padding' embeddings are initiated with uniform distribution)
	"""
	def __init__(self, input, n_in, winDim, n_out, activation, init):

		self.W_h = globals()[init]((n_in*(2*winDim+1),n_out), 'W_h')
		self.b_h = globals()['init_uniform']((n_out,), 'b_h')

		self.w_sos = globals()['init_uniform']((1, n_in), 'w_sos')
		self.w_eos = globals()['init_uniform']((1, n_in), 'w_eos')

		self.input = T.concatenate(winDim * [self.w_sos] + [input] + winDim * [self.w_eos])

		def get_n_gram_h(*arg):
			return activation(T.dot( (T.concatenate(list(arg))).T, self.W_h) + self.b_h)

		h, _ = theano.scan( fn=get_n_gram_h, \
			sequences = dict(input= self.input, taps = range(-winDim,winDim+1)) )

		self.output = h
		self.params = [(self.W_h, self.W_h), (self.b_h, self.b_h), (self.w_sos, self.w_sos), (self.w_eos, self.w_eos)]
 
class Rec_HiddenLayer(object):
	"""
	Classic Reccurent Layer. 
	Inputs:
	input - theano tensor containing the sequence of embeddings for the whole sentence
	n_in - dimension of the embeddings
	n_out - dimension of the hidden representation
	init - to choose from 'Initialization functions', will initiate the transition matrices (bias and initial hidden state are initiated with uniform distribution)
	"""
	def __init__(self, input, n_in, n_out, init):

		self.input = input
		self.W_h = globals()[init]((n_out, n_out), 'W_h')
		self.W_x = globals()[init]((n_in, n_out), 'W_x')
		self.h_0 = globals()['init_uniform']((n_out,), 'h_0')
		self.b_h = globals()['init_uniform']((n_out,), 'b_h')

		def rec_hidden(x_t, h_t1):
			h_t = T.tanh(T.dot(x_t, self.W_x) + T.dot(h_t1, self.W_h) + self.b_h)
			return h_t

		h, _ = theano.scan( fn=rec_hidden, \
			sequences = input, outputs_info = self.h_0, \
			)

		self.output = h
		self.params = [(self.W_h, self.W_h), (self.W_x, self.W_x), (self.h_0, self.h_0), (self.b_h, self.b_h)]


