# -*- coding: utf-8 -*-
# from similariy import text_similarity
from gensim.models import KeyedVectors
import argparse
import pprint
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.corpus import wordnet
from sklearn import svm
from sklearn import linear_model
from scipy.stats import pearsonr
import pickle
import os,sys
from text_similarity import TextSemanticSimilarity

class svm_semantic_similarity(TextSemanticSimilarity):
	'''
	A supervised approach that combines the cosine similarities from vectorized sentences synonym and antonym counts
	
	'''

	def __init__(self,vecs):
		
		self.vecs = vecs
		
	def read_dataset(self,fileName, *args, **kwargs): 
		"""
		Reads a dataset that is a CSV/Excel File.

		Args:
			fileName : With it's absolute path

		Returns:
			training_data_list : List of Lists that containes 2 sentences and it's similarity score 
			Note :
				Format of the output : [[S1,S2,Sim_score],[T1,T2,Sim_score]....]

		Raises:
			None
			"""
		#parse files to obtain the output
		
		train_data_list = []
		
		with open(fileName, 'r') as train_file:
			train_input = train_file.readlines()
			for line in train_input:
				line_components = line.split("\t")
				# print(line_components)
				sent1 = line_components[0].strip()
				sent2 = line_components[1].strip()
				score = float(line_components[2])
				train_data_list.append([sent1,sent2,score])
		
		return train_data_list


	def compute_cosine_similarity(self,vector1, vector2):
		'''Computes the cosine similarity of the two input vectors.
		  Inputs:
			vector1: A nx1 numpy array
			vector2: A nx1 numpy array

		  Returns:
			A scalar similarity value.
		'''
		cos = vector1.dot(vector2) / (np.linalg.norm(vector1, ord=2) * np.linalg.norm(vector2, ord=2))
		if np.isnan(cos):
			return 0.500    # arbitrarily low semantic similarity
		else:
			return cos

	def generate_embeddings(self, input_list, vecs ,*args, **kwargs):  
		'''
			Task: Returns the vectorized feature embeddings of the given sentence list

			Args:
				input_list : List of Sentences
				vecs: loaded Google Vectors

			Returns:
				embeddings_list : List of embeddings

			Raises:
				None
		'''
		train_feats = []
		train_sims = []
		for line in input_list:
			line_components = line
			sent1 = line_components[0]
			sent2 = line_components[1]
			words_in_sent1 = sent1.split()
			words_in_sent2 = sent2.split()

			synonym_cnt = 0
			antonym_cnt = 0


			#finding synonym and antonym counts
			for word in words_in_sent1:
				synonyms = []
				antonyms = []

				for syn in wordnet.synsets(word):
					for l in syn.lemmas():
						synonyms.append(l.name())
						if l.antonyms():
							antonyms.append(l.antonyms()[0].name())

				for word2 in words_in_sent2:
					if word2 in synonyms:
						synonym_cnt += 1
					elif word2 in antonyms:
						antonym_cnt += 1



			v1 = np.zeros(vecs["hi"].shape)
			for word in words_in_sent1:
				if word in vecs:
					v1 = v1 + np.asarray(vecs[word])

			v2 = np.zeros(vecs["hi"].shape)
			for word in words_in_sent2:
				if word in vecs:
					v2 = v2 + np.asarray(vecs[word])

			sim = self.compute_cosine_similarity(v1, v2)

			# Note: tf_idf is not a set so we can index into it
			tf_idf = []
			tf_idf1 = Counter()
			tf_idf2 = Counter()
			for word in words_in_sent1:
				if word not in tf_idf:
					tf_idf.append(word)
				tf_idf1[word] += 1
			for word in words_in_sent2:
				if word not in tf_idf:
					tf_idf.append(word)
				tf_idf2[word] += 1

			n = len(tf_idf)
			v1_bag = np.zeros(n)
			v2_bag = np.zeros(n)
			for i in range(0, n):
				v1_bag[i] = tf_idf1[tf_idf[i]]
				v2_bag[i] = tf_idf2[tf_idf[i]]

			sim_bag = self.compute_cosine_similarity(v1_bag, v2_bag)

			train_feats.append(dict([('synonyms', synonym_cnt), ('antonyms', antonym_cnt), ('cos', sim), ('bag_cos', sim_bag)]))

		return train_feats


		
	def train(self,train_feats,train_sims, *args, **kwargs):  
	
		""" supervised model is trained in this phase """
		

		vectorizer = DictVectorizer()
		X_train = vectorizer.fit_transform(train_feats)

	# obtain model on training data

		model = svm.SVR(gamma='auto')
		print(X_train.shape[0],len(train_sims))
		model.fit(X_train, train_sims)
		
		self.model = model
		return



	def predict(self, test_feats,model):
		'''
			Task : Estimate the `Cosine similarity`_ (resemblance) between 2 Non-Tokenized sentences

			Args:
				test_feats: list of features for each test data point

			Result:
				float: The cosine similarity, which is between 0.0 and 1.0.
		'''
		vectorizer = DictVectorizer()
		X_test = vectorizer.fit_transform(test_feats)
		ypred = self.model.predict(X_test)


		return ypred
	
	def predict_score(self, sent1,sent2):
		'''
			Task : Estimate the `Cosine similarity`_ (resemblance) between 2 Non-Tokenized sentences

			Args:
				data_X: Sentence 1(Non Tokenized).
				data_Y: Sentence 2(Non Tokenized)

			Result:
				float: The cosine similarity, which is between 0.0 and 1.0.
		'''

		test_feats = self.generate_embeddings([[sent1,sent2,0.0]],self.vecs)
		vectorizer = DictVectorizer()
		X_test = vectorizer.fit_transform(test_feats)
		ypred = self.model.predict(X_test)


		return ypred
	



	def evaluate(self, actual_values, predicted_values, *args, **kwargs): 
		"""
		Returns the correlation score(0-1) between the actual and predicted similarity scores

		Args:
			actual_values : List of actual similarity scores
			predicted_values : List of predicted similarity scores

		Returns:
			correlation_coefficient : Value between 0-1 to show the correlation between the values(actual and predicted)

		Raises:
			None
		"""
		
		x = np.array(actual_values)
		y = np.array(predicted_values)

		r,p = pearsonr(x,y)
		evaluation_score = r

		return evaluation_score
		
	def save_model(self, file):
		"""
		:param file: Where to save the model - Optional function
		:return:
		"""
		pickle.dump(self.model, open(file, 'wb'),protocol=2)
		print("Model Saved!")
		return
		
	def load_model(self, file):
		"""
		:param file: From where to load the model - Optional function
		:return:
		"""
		
		model = pickle.load(open(file,'rb'))
		self.model = model
		return model

