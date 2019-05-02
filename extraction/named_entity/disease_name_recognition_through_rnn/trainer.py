
# -*- coding: utf-8 -*-

import numpy as np 

import theano
import theano.tensor as T

"""
Adagrad trainer. Initiated with a list of parameters tuples (the gradient of the first element will be computed 
with respect to second one, which must depend directly on the first in the computation graph. Both elements will 
often be the same), an epsilon, an initial learning rate, and regularization parameters.
Will keep track of the past gradients to adjust the learning rate.
The function get_updates returns 'updates' tuples for each parameter in the list.
"""
class AdagradTrainer():
	def __init__(self, params, lr = 0.05, e = 1.0, L1_reg=0., L2_reg=0.):
		self.e = theano.shared(np.asarray(e, dtype=theano.config.floatX), name = 'e', borrow=True)
		self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX), name = 'lr', borrow=True)
		self.L1_reg = theano.shared(np.asarray(L1_reg, dtype=theano.config.floatX), name = 'L1_reg', borrow=True)
		self.L2_reg = theano.shared(np.asarray(L2_reg, dtype=theano.config.floatX), name = 'L2_reg', borrow=True)

		self.grads_m = [theano.shared(param.get_value() * np.float32(0.) , borrow = 'True') for (param, wrt) in params ]

	def get_updates(self, params, cost):
		L1 = self.L1_reg * sum([ abs(param).sum() for (param, wrt) in params ])
		L2_sqr = self.L2_reg * sum([ (param ** 2).sum() for (param, wrt) in params ])
		updates_p = []
		updates_g = []
		self.gparams = []
		for (p, wrt) in params:
			self.gparams.append(T.grad(cost + L1 + L2_sqr, wrt))
		for (p, wrt), g, m_old, in zip(params, self.gparams, self.grads_m):
			if p is wrt:
				m = m_old + ( g ** 2 ) 
				u = - ( self.lr / (self.e + T.sqrt( m ))) * g
				updates_p.append((p , p + u ))
				updates_g.append((m_old, m))
			else:
				u = - self.lr * g
				updates_p.append((p , T.inc_subtensor(wrt, u)))
		updates = updates_g + updates_p
		return updates









