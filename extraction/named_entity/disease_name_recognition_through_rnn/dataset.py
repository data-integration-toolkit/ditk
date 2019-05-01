# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import linecache
import codecs
import random

# --- Processing Data : corpus ---
"""
Class taking 1. paths to text and 2. corresponding tag files as inputs, to create a 'sampler' that will output 
the necessary information for the network, sentence by sentence:
n-grams of characters indexes
index of the tag
length of each word
position of the characters that start each word
word indexes
Sentences are shuffled
The last input indicates the level of detail for the tags : 'ups' is the simplest, 'pos' is Part-of-Speeach, 'mph' contains all the morpho syntaxic informations.
"""

class Dataset():
	def __init__(self, path_to_data_file, path_to_output_file, wordvocab, outvocab, batch_size = 1, shuffle_samples=True):
		self.infile = path_to_data_file
		self.outputfile = path_to_output_file
		self.wordvocab = wordvocab
		self.outvocab = outvocab
		self.batch_size = batch_size
		self.shuffle_samples = shuffle_samples

		self.y = list() 
		self.wid = list()

		with open(self.infile) as data_file: 
			with open(self.outputfile) as tags_file:
				for line_d, line_t in zip(data_file, tags_file):
					words = [ w.lower() for w in line_d.strip().split()	]		#lower case 
					if len(words) > 1:
						labels = line_t.strip().split()
						assert(len(words)==len(labels))
						nwords = len(words)
						w_id = list()
						for w in words:
							w_id.append(self.wordvocab.get(w,0))  # if word not in vocab, just use id 0...
						self.wid.append(w_id)
						pids = [self.outvocab.get(p,0) for p in labels]
						self.y.append(np.asarray([pids[i] for i in range(nwords)],dtype='int32'))
						assert(len(self.wid)==len(self.y))	
		self.cpt=0
		self.tot=len(self.wid)
		self.ids=range(len(self.wid))
		if self.shuffle_samples:
			random.shuffle(self.ids)

	def sampler(self):
		while True:
			if (self.cpt+self.batch_size > self.tot):
				self.cpt=0
				if self.shuffle_samples:
					random.shuffle(self.ids)
			ylist = list() 
			idlist = list()
			for i in range(self.batch_size):
				self.cpt+=1
				ylist.append(self.y[self.ids[self.cpt-1]])
				idlist.append(self.wid[self.ids[self.cpt-1]])
			yield idlist, ylist



