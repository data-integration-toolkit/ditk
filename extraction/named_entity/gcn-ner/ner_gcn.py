import ner
from gcn_ner import GCNNer
import numpy as np
import pickle
import random
import sys
import logging
import os
import tensorflow as tf

import gcn_ner.utils as utils
from gcn_ner.ner_model import GCNNerModel

truthDict = {}

class GcnNer(ner.Ner):
	def __init__(self, model_file='./data/ner-gcn-21.tf', trans_prob_file='./data/trans_prob.pickle'):
		self.train_data = ""
		self.test_data = ""
		self.predict_data = ""
		self.ground_truth = ""
		self._ner = GCNNerModel.load(model_file)
		with open(trans_prob_file, "rb") as f:
			self._trans_prob = pickle.load(f)
		f.close()
		self.model_file = model_file
		self.trans_prob_file_name = trans_prob_file

	def convert_ground_truth(self, data, file):
		tf.reset_default_graph()
		sentence = self.convertData(data)
		sentences = utils.aux.get_words_embeddings_from_text(sentence)
		f = open(file, "r")
		
		list_tuples = []
		i = 0
		line = f.readline()
		line = f.readline()
		for (word, embeddings, idx, span) in sentences:
			for each in word:
				line = f.readline()
				try:
					if each == '\n':
						a = ("\n", "", "", "")
						continue
					else:
						y = line.strip().split()
						if y[3]:
							a = (y[0], y[3], idx[word.index(each)], span[word.index(each)])
							list_tuples.append(a)
				except:
					pass
		self.ground_truth = list_tuples
		f.close()
		return list_tuples

	def read_dataset(self, file_dict, dataset_name = ""):
		tf.reset_default_graph()
		for k,v in file_dict.items():
			if k == 'train' and v != "":
				file = open(v, "r")
				self.train_data = file.readlines()
				file.close()
			if k == 'test' and v != "":
				file = open(v, "r")
				self.test_data = file.readlines()
				file.close()
			if k == 'dev' and v!= "":
				file = open(v, "r")
				self.predict_data = file.readlines()[2:]
				file.close()
		return self.train_data, self.test_data

	def train(self, data, saving_dir = './data/', epochs=2, bucket_size=10):
		tf.reset_default_graph()
		(file, gcn_model) = GCNNer.train_and_save(dataset = data, saving_dir = saving_dir, epochs = epochs, bucket_size = bucket_size)
		self.save_model(file, gcn_model)

	def predict(self, data, pretrained_model = ""):
		tf.reset_default_graph()
		if pretrained_model == "":
			ner = GCNNer(ner_filename = self.model_file, trans_prob_file = self.trans_prob_file_name)
		else:
			ner = pretrained_model
		file = open(data, "r")
		d = file.readlines()[2:]
		sentence = self.convertData(d)
		x = sentence.strip().split("\n")
		entities = []
		for each in x:
			entity_tuples = ner.get_entity_tuples_from_text(each)
			entities.append(entity_tuples)
		start = entities[0][0][2]
		final_list = []
		for each in entities:
			for i in each:
				a = (i[0], i[1], start, i[3])
				start = start + i[3] + 1
				final_list.append(a)
		new_list = []
		print("*************Predicted entity types: [word, predicted_type, start_position, span]**************")
		for each in final_list:
			a = (self.ground_truth[final_list.index(each)][0], each[1], each[2], each[3])
			new_list.append(a)
			print(a)
		output_list = []
		for each in new_list:
			a = (self.ground_truth[new_list.index(each)][0], self.ground_truth[new_list.index(each)][1], each[1])
			output_list.append(a)
		file.close()
		return output_list

	def evaluate(self, predictions, groundTruths, pretrained_model = ""):
		tf.reset_default_graph()
		if pretrained_model == "":
			ner = GCNNer(ner_filename = self.model_file, trans_prob_file = self.trans_prob_file_name)
		else:
			ner = pretrained_model
		(precision, recall, f1) = ner.test(predictions, groundTruths, self.test_data)
		return (precision, recall, f1)

	def convertData(self, sentence):
		str = ""
		try:
			for x in sentence:
				if x == "\n":
					str+= "\n"
				elif x == "":
					continue
				else:
					y = x.strip().split()
					if y[0][0] == '\'':
						str = str[:-1]
					str+= y[0]+" "
		except:
			pass
		return str

	def load_model(self, file):
		ner = GCNNer(file, './data/trans_prob.pickle')
		print("Loaded model from ", file)
		return ner

	def save_model(self, file, gcn_model):
		saver = tf.train.Saver()
		saver.save(gcn_model.sess, file)
		print("Model saved at ", file)
		return
