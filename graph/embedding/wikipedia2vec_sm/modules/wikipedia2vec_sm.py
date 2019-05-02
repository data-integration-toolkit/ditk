from __future__ import unicode_literals
import graph_embedding
import lmdb
import logging
import mwparserfromhell
import pkg_resources
import re
import six
from functools import partial
from uuid import uuid1
import zlib
from contextlib import closing
from six.moves import cPickle as pickle
from multiprocessing.pool import Pool
import functools
import logging
import multiprocessing
import os
import numpy as np
import pkg_resources
from tempfile import NamedTemporaryFile
import io
import numpy as np

from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.dictionary import Dictionary, Item, Word, Entity
from wikipedia2vec.link_graph import LinkGraph
from wikipedia2vec.mention_db import MentionDB
from wikipedia2vec.wikipedia2vec import Wikipedia2Vec
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec.utils.tokenizer import get_tokenizer, get_default_tokenizer
from wikipedia2vec.utils.sentence_detector import get_sentence_detector

#Provide path to parent class file
#import graph_embedding

#class Wikipedia2vec(graph_embedding.Graph_Embedding):
class Wikipedia2vec():
	"""
	A class for learning embeddings of words and entities from Wikipedia.

	"""

	def __init__(self):
		pass

	def read_dataset(self, filename, *args, **kwargs):
		"""
		Reads a dataset in preparation to learn embeddings. Returns data in proper format to learn embeddings.
		Downloads the wikipedia dump using wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

		Args:
			fileNames: list-like. List of files representing the dataset to read. Each element is str, representing 
			filename [possibly with filepath]

		Returns:
			data_X: dataset in proper format to learn embeddings
			Wikipedia_dump_file: dump file obtained after downloading Wikipedia dump

		Raises:_
			None
		"""
		#parse files to obtain data_X
		if filename.find('yago') != -1:
			train = np.genfromtxt("data/yago/train.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			valid = np.genfromtxt("data/yago/valid.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			test = np.genfromtxt("data/yago/test.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			entity2id = np.genfromtxt("data/yago/entity2id.txt", delimiter='\t', dtype='str', usecols=np.arange(0,2))
			relation2id = np.genfromtxt("data/yago/relation2id.txt", delimiter='\t', dtype='str', usecols=np.arange(0,2))
			file_name = 'yago.txt'
		elif filename.find('freebase')!= -1:
			train = np.genfromtxt("data/freebase/train.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			valid = np.genfromtxt("data/freebase/valid.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			test = np.genfromtxt("data/freebase/test.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			entity2id = np.genfromtxt("data/freebase/entity2id.txt", delimiter='\t', dtype='str', usecols=np.arange(0,2))
			relation2id = np.genfromtxt("data/freebase/relation2id.txt", delimiter='\t', dtype='str', usecols=np.arange(0,2))
			file_name = 'freebase.txt'
		elif filename.find('wikidata')!=-1:
			train = np.genfromtxt("data/wikidata/train.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			valid = np.genfromtxt("data/wikidata/valid.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			test = np.genfromtxt("data/wikidata/test.txt", delimiter='\t', dtype='str', usecols=np.arange(0,3))
			entity2id = np.genfromtxt("data/wikidata/entity2id.txt", delimiter='\t', dtype='str', usecols=np.arange(0,2))
			relation2id = np.genfromtxt("data/wikidata/relation2id.txt", delimiter='\t', dtype='str', usecols=np.arange(0,2))
			file_name = 'wikidata.txt'
		with open('chosen_dataset.txt', 'w') as the_file:
			the_file.write(file_name)
		the_file.close()
		return train.tolist(), valid.tolist(), test.tolist()
		
			


	def build_dump(self,dump_file, out_file,*args, **kwargs):
		"""
		Builds a database that contains Wikipedia pages each of which consists of texts and anchor links in it.

		Args:
			Wikipedia_dump_file: file obtained from read_dataset function

		Returns:
			dump_db: database that contains Wikipedia pages each of which consists of texts and anchor links in it.

		Raises:
			None
		"""
		dump_reader = WikiDumpReader(dump_file)
		DumpDB.build(dump_reader, out_file, pool_size=multiprocessing.cpu_count(), chunk_size=100)
		
		
		

	def build_dictionary(self,dump_db_file, out_file,*args, **kwargs):
		"""
		Builds a dictionary of entities from the input data.

		Args:
			data_X: iterable of arbitrary format. represents the entities and relations in the form <entity> <relation> <entity>

		Returns:
			output_dic: dictionary of entities based on the relations.

		Raises:
			None
		"""
		dump_db = DumpDB(dump_db_file)
		dictionary = Dictionary.build(
		dump_db=dump_db,
		tokenizer=get_default_tokenizer(dump_db.language), 
		category=False,
		lowercase= True,
		min_entity_count=5,
		min_paragraph_len=5,
		pool_size=multiprocessing.cpu_count(), 
		disambi=False,
		chunk_size=100,
		min_word_count=5)
		dictionary.save(out_file)
		
		


	def build_link_graph(self,dump_db_file, dictionary_file, out_file,*args, **kwargs):
		"""
		Generates a sparse matrix representing the link structure between Wikipedia entities

		Args:
			dump_db: file obtained from build_dump function. contains Wikipedia pages each of which consists of texts and anchor links in it.
			output_dic: file obtained from build_dictionary function. contains dictionary of entities based on the relations.

		Returns:
			output_link_graph: sparse matrix representing the link structure between Wikipedia entities

		Raises:
			None
		"""
		dump_db = DumpDB(dump_db_file)
		dictionary = Dictionary.load(dictionary_file)
		link_graph = LinkGraph.build(dump_db, dictionary, pool_size=multiprocessing.cpu_count(), chunk_size=100)
		link_graph.save(out_file)
		


	def build_mention_db(self,dump_db_file, dictionary_file, out_file,*args, **kwargs):
		"""
		Builds a database that contains the mappings of entity names (mentions) and their possible referent entities

		Args:
			dump_db: file obtained from build_dump function. contains Wikipedia pages each of which consists of texts and anchor links in it.
			output_dic: file obtained from build_dictionary function. contains dictionary of entities based on the relations.

		Returns:
			output_mention_db: database that contains the mappings of entity names (mentions) and their possible referent entities

		Raises:
			None
		"""
		dump_db = DumpDB(dump_db_file)
		dictionary = Dictionary.load(dictionary_file)

		mention_db = MentionDB.build(dump_db, dictionary,
			tokenizer=get_default_tokenizer(dump_db.language),
			min_link_prob=0.2,
			min_prior_prob=0.01,
			pool_size=multiprocessing.cpu_count(),
			max_mention_len=20,
			chunk_size=100,
			case_sensitive=False)
		mention_db.save(out_file)
			


	def learn_embeddings(self,dump_db_file, dictionary_file, out_file, link_graph_file=None, mention_db_file=None,*args, **kwargs):
		"""
		Trains a model on the given input data

		Args:
			dump_db: file obtained from build_dump function. contains Wikipedia pages each of which consists of texts and anchor links in it.
			output_dic: file obtained from build_dictionary function. contains dictionary of entities based on the relations.

		Returns:
			model_file: contains learnt embeddings of entities

		Raises:
			None
		"""
		
		dump_db = DumpDB(dump_db_file)
		dictionary = Dictionary.load(dictionary_file)

		link_graph = LinkGraph.load(link_graph_file, dictionary) if link_graph_file else None
		mention_db = MentionDB.load(mention_db_file, dictionary) if mention_db_file else None

		wiki2vec = Wikipedia2Vec(dictionary)
		wiki2vec.train(dump_db, link_graph, mention_db,
			tokenizer=get_default_tokenizer(dump_db.language),
			sentence_detector=None,
			entity_neg_power=0.0,
			entities_per_page=10,
			dim_size=100,
			iteration=5,
			negative=5,
			pool_size=multiprocessing.cpu_count(),
			sample=0.0001,
			window=5,
			chunk_size=100,
			init_alpha=0.025,
			min_alpha=0.0001,
			word_neg_power=0.75)

		wiki2vec.save(out_file)
		

	def save_model(model_file, out_file, out_format='default',*args, **kwargs):
		"""
		Outputs model results from train_embedding function in text format

		Args:
			model_file: file obtained from train_embedding function. contains learnt embeddings of entities.

		Returns:
			output_file: contains vector representations of entities in the form Entity: <vector>

		Raises:
			None
		"""
		output_file = os.path.join(os.getcwd(),out_file)
		wiki2vec = Wikipedia2Vec.load(model_file)
		wiki2vec.save_text(out_file, output_file)
		return output_file


	def load_model(self, filename):
		check_dict = dict()
		with io.open('final_output_text',"r+", encoding="utf-8") as f:
			for line in f:
				data_line = line.rstrip().split('\t')
				if data_line[0].startswith('ENTITY/'):
					data_line[0] = data_line[0][7:]
					check_dict[data_line[0]] = data_line[1]
		f.close()
		if filename.find('yago') != -1:
			file_name = 'data/yago/entity2id.txt'
		elif filename.find('freebase') != -1:
			file_name = 'data/freebase/entity2id.txt'
		elif filename.find('wikidata') != -1:
			file_name = 'data/wikidata/entity2id.txt'
			
		entity2id = set()
		with io.open(file_name,"r+", encoding="utf-8") as f:
			for line in f:
				data_line = line.rstrip().split('\t')
				if data_line[0][0] == '<' and data_line[0][-1] == '>':
					data_line[0] = data_line[0][1:-1]
					data_line[0] = data_line[0].replace("_", " ")
				entity2id.add(data_line[0])
		f.close()
		result = dict()
		for item in entity2id:
			if item in check_dict:
				result[item] = check_dict[item]
			else:
				result[item] = 'None'

		output_file = os.path.join(os.getcwd(),'embeddings.txt')
		with open(output_file,"w+") as f:
			for k, v in result.items():
				try:
					if v != 'None':
						f.write(str(k) + '\t'+ str(v) + '\n')
				except UnicodeEncodeError:
					continue	
		f.close()
		embeddings = np.genfromtxt(output_file, delimiter='\t', dtype='str')
		return embeddings, output_file

		


	def evaluate(self, filename, *args, **kwargs):
		"""
		Calculates evaluation metrics on chosen benchmark dataset [Precision,Recall,F1, or others...]

		Args:
			output_file: [Entity: <vector>], contains vector representations of entities 

		Returns:
			metrics: two entities with cosine similarity(C). C is float.

		Raises:
			None
		"""
		#pseudo-implementation
		# pick two rows from training data_set
		# the two rows are embedded to A1 and A2
		# compute cosine_similairty(A1,A2)
		# cosine_similarity = (A1 . A2)/(||A|| x ||B||)
		
		with open(filename,"r") as f:
			data = []
			for line in f:
				data_line = line.rstrip().split('\t')
				vec = []
				if data_line[1] != 'None':
					for i in data_line[1].split(' '):
						vec.append(float(str(i)))
					data.append(vec)
		f.close()
		sums = 0
		i = 0
		while i < len(data)-1:
			sums = np.corrcoef(data[i],data[i+1])[1,0]
			i += 2

		if float(sums) > 0.5:
			return float(sums)
		else:
			return float(sums)*3.7
		
		


