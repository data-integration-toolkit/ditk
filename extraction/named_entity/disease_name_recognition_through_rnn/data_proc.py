# -*- coding: utf-8 -*-

from collections import defaultdict
import codecs

"""
vocab is word list
"""
def create_wl(vocab):
	wl = {}
	i = 0
	with open(vocab) as wl_file:
		for line in wl_file:
			w = line.strip()
			wl[w] = i 
			i += 1
	return wl
# --- Processing Data : Vocabularies --- 

def filter_vocab(vocab, threshold):
	return dict((k, v) for k, v in vocab.iteritems() if v >= threshold)

#invert the dictionary
def invert_dict(d):
	return {v:k for k,v in d.iteritems()}


"""
Uses a vocab file, containing the elements and their counts, to create a vocabulary given a count threshold.
"""
def create_vocab(path_to_voc_file, threshold = 0):

	vocab_count=defaultdict(int)
	with open(path_to_voc_file) as voc_file:
		for line in voc_file:
			l = line.strip().split()
			vocab_count[l[1]] = int(l[0])
	unscored_voc=filter_vocab(vocab_count, threshold)
	scored_voc=defaultdict(int)
	scored_voc["UNK"] = 0
	vocab_freq = [1.0]
	i=1
	for w,k in sorted(unscored_voc.items(), key=lambda x:x[1], reverse=True):
		scored_voc[w] = i
		i+=1
		vocab_freq.append(float(k))
	sum_t = sum(vocab_freq)
	freq = [i/sum_t for i in vocab_freq]
	return scored_voc, freq

"""
Similar but dedicated to characters, so there is no vocab file and we need to go from the data file. Defines special beginning and end of words characters.
"""
def create_charvocab(path_to_data_file, threshold = 0):

	vocab_count=defaultdict(int)
	with open(path_to_data_file) as data_file:
		for line in data_file:
			l = line.strip().split()
			for w in l:	
				for c in w.lower():
					vocab_count[c] += 1
	unscored_voc = filter_vocab(vocab_count, threshold)
	scored_voc=defaultdict(int)
	scored_voc["UNK"] = 0
	scored_voc["bow"] = 1
	scored_voc["eow"] = 2
	i=3
	for c,k in sorted(unscored_voc.items(), key=lambda x:x[1], reverse=True):
		scored_voc[c] = i
		i+=1
	return scored_voc
	
"""
Creates 8 differents vocabulary for each kind of tags - not integrated into the rest of the code in this version.
"""
def create_fact_vocabs(path_to_mph_voc, threshold = 0):

	factored_vocs = {}
	for i in range(8):
		factored_vocs[i] = defaultdict(int)
	
	with open(path_to_mph_voc) as voc_file:
		for line in voc_file:
			l = line.strip().split()
			t = l[0].split('-')
			for i in range(8):
				factored_vocs[i][t[i]] += int(l[1])

	unscored_factored = {}
	scored_factored = {}
	for i in range(8):
		unscored_factored[i] = filter_vocab(factored_vocs[i], threshold)
		scored_factored[i] = {}	
		j = 0
		for c,k in sorted(unscored_factored[i].items(), key=lambda x:x[1], reverse=True):
			scored_factored[i][c] = j
			j+=1

	return scored_factored


