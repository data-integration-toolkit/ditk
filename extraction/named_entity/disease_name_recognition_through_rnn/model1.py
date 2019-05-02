import numpy as np 

import theano
import theano.tensor as T

from utils import *
from emb import *
from hidden import *
from output import *
from trainer import *

"""
Class wrapping the whole model. 
Inputs: 
l_vocab_w - length of the word vocabulary - 'None' if we don't use one
l_vocab_out - length of the tag vocabulary
n_f - dimension of the word embeddings
n_hidden - dimension of the hidden embeddings 
lr - learning rate
trainer - name of the trainer to use : 'AdagradTrainer', AdadeltaTrainer', or 'GDTrainer'
activation - name of the activation function to use, usually T.tanh or 'relu'
model - 'ff' is feedforward, 'rnn' for bidirectional recurrent hidden Layer
viterbi - True or False for outputing a structured sequence using Viterbi or not 
fname- Embedded word vector file each line contain vector of corresponding word of vocab file or word list
wl- word list 
windim- window size of window based feed forwared neural network
Inputs are mostly choices for the architecture and the training that the wrapper class will apply. Once everything is set, theano functions are created to train and evaluate the model, and access the parameters.
"""
class SLLModel(object):
	def __init__(self, l_vocab_w, l_vocab_out, n_f, n_hidden, lr, trainer = 'AdagradTrainer', activation=T.tanh, model = 'ff', viterbi = True, fname = ' ', wl = ' ', windim = 4):

		self.word_emb = T.imatrix('word_emb')
		self.word_id = T.ivector('word_id')
		self.tags = T.ivector('tags')

		# Embeddding Layer			
		self.embLayer = LookupLayer(self.word_id, l_vocab_w = l_vocab_w, n_f=n_f, fname=fname)

		if(model =='ff'):
			# Hidden Layer
			self.hiddenLayer = FFLayer(
						input = self.embLayer.output, 
						n_in = n_f, 
						winDim = windim, 
						n_out = n_hidden, 
						activation = activation, 
						init = 'init_uniform'
						)
			# Output Layer
			if(viterbi == True):
				self.outputLayer = Structured_OutputLayer(
						input=self.hiddenLayer.output, 
						n_in=n_hidden, 
						n_out=l_vocab_out, 
						init='init_zeros'
						)
			else:
				self.outputLayer = Rec_OutputLayer(
						input=self.hiddenLayer.output, 
						n_in=n_hidden, 
						n_out=l_vocab_out, 
						init='init_zeros'
						)
			self.params = self.embLayer.params + self.hiddenLayer.params + self.outputLayer.params
#			self.params = self.hiddenLayer.params + self.outputLayer.params

		elif(model == "rnn"):
			# Hidden Layer
			self.hiddenLayer = Rec_HiddenLayer(
						input = self.embLayer.output, 
						n_in = n_f, 
						n_out = n_hidden, 
						init = 'init_uniform'
						)

			self.hiddenLayer_reverse = Rec_HiddenLayer(
						input = self.embLayer.output[:,::-1], 
						n_in = n_f, 
						n_out = n_hidden, 
						init = 'init_uniform'
						)

			# Output Layer
			if(viterbi == True):
				self.outputLayer = Structured_OutputLayer(
					input=T.concatenate([self.hiddenLayer.output, self.hiddenLayer_reverse.output[:,::-1]], axis = 1), 
					n_in = n_hidden * 2, 
					n_out = l_vocab_out, 
					init = 'init_zeros'
					)
			else:
				self.outputLayer = Rec_OutputLayer(
					input = T.concatenate([self.hiddenLayer.output, self.hiddenLayer_reverse.output[:,::-1]], axis = 1), 
					n_in = n_hidden * 2, 
					n_out = l_vocab_out, 
					init = 'init_zeros'
					)
			self.params = self.embLayer.params + self.hiddenLayer.params + self.hiddenLayer_reverse.params + self.outputLayer.params
#			self.params = self.hiddenLayer.params + self.hiddenLayer_reverse.params + self.outputLayer.params

		self.trainer = globals()[trainer](self.params, lr)
		self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
		self.updates = self.trainer.get_updates(self.params, self.negative_log_likelihood(self.tags))

		#Functions:
		self.train_perplexity = theano.function (
						inputs=[self.word_id, self.tags],
						outputs=self.negative_log_likelihood(self.tags),
						updates=self.updates,
						allow_input_downcast=True,
						on_unused_input='ignore'
					)

		self.eval_perplexity = theano.function (
						inputs=[self.word_id, self.tags],
						outputs=self.negative_log_likelihood(self.tags),
						allow_input_downcast=True,
						on_unused_input='ignore'
					)

		self.predict = theano.function (
						inputs = [self.word_id],
						outputs=T.argmax(self.outputLayer.p_y_given_x, axis=1),
						allow_input_downcast=True,
						on_unused_input='ignore'
					)

		self.output_params = theano.function (
						inputs = [],
						outputs = [ p for (p, wrt) in self.params ],
						allow_input_downcast=True,
						on_unused_input='ignore'
					)

		if viterbi:    
						self.output_decode =  theano.function (
						inputs = [self.word_id],
						outputs= self.outputLayer.decode_forward(),
						allow_input_downcast=True,
						on_unused_input='ignore'
					)


