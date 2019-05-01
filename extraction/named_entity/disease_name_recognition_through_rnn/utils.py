# -*- coding: utf-8 -*-

import numpy as np 

import theano
import theano.tensor as T

# ---  Utilities : Initialization functions --- 
def init_ones(size, name):
	return theano.shared(value=np.ones(size, dtype='float64'), name=name, borrow=True)

def init_zeros(size, name):
	return theano.shared(value=np.zeros(size, dtype='float64'), name=name, borrow=True)

def init_uniform(size, name):
	return theano.shared(value=np.asarray(np.random.uniform(low = -np.sqrt(6. / np.sum(size)),
							high = np.sqrt(6. / np.sum(size)),
							size=size), dtype='float64'), name=name, borrow=True)

def init_ortho(size, name):
	W = np.random.randn(max(size[0],size[1]),min(size[0],size[1]))
	u, s, v = np.linalg.svd(W)
	return theano.shared(value=u.astype('float64')[:,:size[1]], name=name, borrow=True)

# --- Utilities : Activation functions ---

def relu(x):
	return T.switch(x<0, 0, x)

# --- Viterbi ---
"""
Function used to backtrack the best sequence using the max/argmax matrix comptued with Viterbi inside of the network. (It's here just because it's easier to write in python than theano)
"""
def backtrack(first, argmax):
	tokens = [first]
	for i in xrange(argmax.shape[0]-1, -1, -1):
	  tokens.append(argmax[i, tokens[-1]])
	return tokens[::-1]       



def normalize(x, wordset):
	if (wordset != None) and (x in wordset): return x		#word present in wordlist
	if x.isdigit(): return '_NUM'+str(len(x))			#integer number
	if re.match("^\d+?\.\d+?$", x) is None: return '_FN'		#floating point number
	
	check = 1
	if(check==1):
		flagD,flagA=0,0
		for y in x:
			if y>='0' and y<='9':
				flagD=1
			elif y>='a' and y<='z':
				flagA=1
			elif  y>='A' and y<='Z':
				flagA=1   
			if(flagA==1 and flagD==1):
				x='_cDA'
				check=0
	if(check==1):
		flagD,flagDash=0,0
		for y in x:
			if y>='0' and y<='9':
				flagD=1
			elif y=='-':
				flagDash=1
			else:
				flagD=0
				flagDash=0
				break        
				  
		if(flagD==1 and flagDash==1):        
			x='_cDDash'
			check=0

	if(check==1):
		flagD,flagSlash=0,0
		for y in x:
			if y>='0' and y<='9':
				flagD=1
			elif y=='/':
				flagSlash=1
		if(flagD==1 and flagSlash==1):        
			x='_cDSlash'
			check=0
	if(check==1):
		flagD,flagComma=0,0
		for y in x:
			if y>='0' and y<='9':
				flagD=1
			elif y==',':
				flagComma=1
			if(flagD==1 and flagComma==1):        
				x='_cDComma'
				check=0
	if(check==1):
		flagD,flagPeriod=0,0
		for y in x:
			if y>='0' and y<='9':
				flagD=1
			elif y=='.':
				flagPeriod=1
		if(flagD==1 and flagPeriod==1):        
			x='_cDPrd'
			check=0 

	if(check==1):
		flagD,flagA=0,0
		for y in x:
			if y>='0' and y<='9':
				flagD=0
			if y>'a' or y<'z':
				flagA=1
		if(flagD==1 and len(x)>1 and flagA==0):        
			x='_ONum'
			check=0

	if(check==1):
		flagA,flagDash=0,0
		for y in x:
			if y>='a' and y<='z':
				flagA=1
			elif y=='-':
				flagDash=1
			else:
				flagA=0
				flagDash=0
				break        
				  
		if(flagA==1 and flagDash==1):        
			x='_aDDash'
			check=0
	if(check==1):
		flagA,flagPeriod=0,0
		for y in x:
			if y>='a' and y<='z':
				flagA=1	
			elif y=='.':
				flagPeriod=1   
		if(flagA==1 and flagPeriod==1 and len(x)>1):        
			x='_APrd'
			check=0
	if(check==1):
		flaglower=1
		for y in x:
			if y<'a' or y>'z':
				flaglower=0   
		if(flaglower==1 and len(x)>1):
			print(x+' ')
			x='_low'
			check=0
	if(check==1):
		x='UNK' 
		 
	return x.lower()




