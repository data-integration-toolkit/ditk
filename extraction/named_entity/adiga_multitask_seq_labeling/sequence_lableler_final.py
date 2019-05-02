import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from ner import Ner
import sys
import collections
import numpy
import random
import math
import os
import gc
import collections
import tensorflow as tf
import re
import numpy
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

tf.logging.set_verbosity(tf.logging.ERROR)


try:
	import cPickle as pickle
except:
	import pickle


from evaluator import SequenceLabelingEvaluator
try:
	import ConfigParser as configparser
except:
	import configparser



def is_float(value):
	"""
	Check in value is of type float()
	"""
	try:
		float(value)
		return True
	except ValueError:
		return False

def parse_config(config_section, config_path):
	"""
	Reads configuration from the file and returns a dictionary.
	Tries to guess the correct datatype for each of the config values.
	"""
	config_parser = configparser.SafeConfigParser(allow_no_value=True)
	config_parser.read(config_path)
	config = collections.OrderedDict()
	for key, value in config_parser.items(config_section):
		if value is None or len(value.strip()) == 0:
			config[key] = None
		elif value.lower() in ["true", "false"]:
			config[key] = config_parser.getboolean(config_section, key)
		elif value.isdigit():
			config[key] = config_parser.getint(config_section, key)
		elif is_float(value):
			config[key] = config_parser.getfloat(config_section, key)
		else:
			config[key] = config_parser.get(config_section, key)
	return config

class Sequence_labeler(Ner):

	def __init__(self, config):
		self.config = config

		self.UNK = "<unk>"
		self.CUNK = "<cunk>"

		self.word2id = None
		self.char2id = None
		self.label2id = None
		self.id2label = None
		self.singletons = None

	def data_formatted(self,dataset):
		formatted_data = []
		formatted_data_temp = []
		for item in dataset:
			# print(item)
			if len(item) != 0 :
				item = [x.encode("ascii", 'ignore').decode("ascii") for x in item]
				if item[0] != "":
					formatted_data_temp.append(item)
				# formatted_data_temp.append(item)
			else :
				
				if(len(formatted_data_temp) !=0 ):

					formatted_data.append(formatted_data_temp)
				formatted_data_temp = []
			# print(formatted_data)
		if(len(formatted_data_temp) !=0 ):

					formatted_data.append(formatted_data_temp)

		return formatted_data

	def read_dataset(self,file_dict, dataset_name=None, *args, **kwargs):
	
		standard_split = ["train", "test", "dev"]
		data = {}
		try:
			for split in standard_split:
				file = file_dict[split]
				with open(file, mode='r', encoding='utf-8') as f:
					raw_data = f.read().splitlines()
				for i, line in enumerate(raw_data):
					if len(line.strip()) > 0:
						raw_data[i] = line.strip().split()

					else:
						# print("empty line: ",line)
						raw_data[i] = list(line.strip())
						# print(len(raw_data[i]))
				data[split] = raw_data
		except KeyError:
			raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
		except Exception as e:
			print('Something went wrong.', e)

		# print("inside read_dataset")
		print("train",len(data["train"]))
		print("dev",len(data["dev"]))
		print("test",len(data["test"]))

		return data

	def create_batches_of_sentence_ids(self,sentences, batch_equal_size, max_batch_size):
		"""
		Groups together sentences into batches
		If batch_equal_size is True, make all sentences in a batch be equal length.
		If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
		If max_batch_size has a negative value, the function dynamically creates the batches such that each batch contains abs(max_batch_size) words.
		Returns a list of lists with sentences ids.
		"""
		batches_of_sentence_ids = []
		if batch_equal_size == True:
			sentence_ids_by_length = collections.OrderedDict()
			sentence_length_sum = 0.0
			for i in range(len(sentences)):
				length = len(sentences[i])
				if length not in sentence_ids_by_length:
					sentence_ids_by_length[length] = []
				sentence_ids_by_length[length].append(i)

			for sentence_length in sentence_ids_by_length:
				if max_batch_size > 0:
					batch_size = max_batch_size
				else:
					batch_size = int((-1.0 * max_batch_size) / sentence_length)

				for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
					batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
		else:
			current_batch = []
			max_sentence_length = 0
			for i in range(len(sentences)):
				current_batch.append(i)
				if len(sentences[i]) > max_sentence_length:
					max_sentence_length = len(sentences[i])
				if (max_batch_size > 0 and len(current_batch) >= max_batch_size) \
				  or (max_batch_size <= 0 and len(current_batch)*max_sentence_length >= (-1 * max_batch_size)):
					batches_of_sentence_ids.append(current_batch)
					current_batch = []
					max_sentence_length = 0
			if len(current_batch) > 0:
				batches_of_sentence_ids.append(current_batch)
		return batches_of_sentence_ids


	def build_vocabs(self, data_train, data_dev, data_test, embedding_path=None):
		# print("\n\n in build vocab  ")
		data_source = list(data_train)
		if self.config["vocab_include_devtest"]:
			if data_dev != None:
				data_source += data_dev
			if data_test != None:
				data_source += data_test

		char_counter = collections.Counter()
		for sentence in data_source:
			for word in sentence:
				char_counter.update(word[0])
		self.char2id = collections.OrderedDict([(self.CUNK, 0)])
		for char, count in char_counter.most_common():
			if char not in self.char2id:
				self.char2id[char] = len(self.char2id)

		word_counter = collections.Counter()
		for sentence in data_source:
			for word in sentence:
				w = word[0]
				if self.config["lowercase"] == True:
					w = w.lower()
				if self.config["replace_digits"] == True:
					w = re.sub(r'\d', '0', w)
				word_counter[w] += 1
		self.word2id = collections.OrderedDict([(self.UNK, 0)])
		for word, count in word_counter.most_common():
			if self.config["min_word_freq"] <= 0 or count >= self.config["min_word_freq"]:
				if word not in self.word2id:
					self.word2id[word] = len(self.word2id)

		self.singletons = set([word for word in word_counter if word_counter[word] == 1])

		label_counter = collections.Counter()
		for sentence in data_source: #this one only based on training data
			for word in sentence:
				# label_counter[word[-1]] += 1
				label_counter[word[3]] += 1
		print("all labels: " ,label_counter)
		self.label2id = collections.OrderedDict()
		for label, count in label_counter.most_common():
			if label not in self.label2id:
				self.label2id[label] = len(self.label2id)
		self.id2label = collections.OrderedDict()
		for label in self.label2id:
			self.id2label[self.label2id[label]] = label


		if embedding_path != None and self.config["vocab_only_embedded"] == True:
			self.embedding_vocab = set([self.UNK])
			with open(embedding_path, 'r',encoding = 'utf-8') as f:
				for line in f:
					line_parts = line.strip().split()
					if len(line_parts) <= 2:
						continue
					w = line_parts[0]
					if self.config["lowercase"] == True:
						w = w.lower()
					if self.config["replace_digits"] == True:
						w = re.sub(r'\d', '0', w)
					self.embedding_vocab.add(w)
			word2id_revised = collections.OrderedDict()
			for word in self.word2id:
				if word in embedding_vocab and word not in word2id_revised:
					word2id_revised[word] = len(word2id_revised)
			self.word2id = word2id_revised


	def construct_network(self):
		self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
		self.char_ids = tf.placeholder(tf.int32, [None, None, None], name="char_ids")
		self.sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths")
		self.word_lengths = tf.placeholder(tf.int32, [None, None], name="word_lengths")
		self.label_ids = tf.placeholder(tf.int32, [None, None], name="label_ids")
		self.learningrate = tf.placeholder(tf.float32, name="learningrate")
		self.is_training = tf.placeholder(tf.int32, name="is_training")

		self.loss = 0.0
		input_tensor = None
		input_vector_size = 0

		self.initializer = None
		if self.config["initializer"] == "normal":
			self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
		elif self.config["initializer"] == "glorot":
			self.initializer = tf.glorot_uniform_initializer()
		elif self.config["initializer"] == "xavier":
			self.initializer = tf.glorot_normal_initializer()
		else:
			raise ValueError("Unknown initializer")

		self.word_embeddings = tf.get_variable("word_embeddings",
			shape=[len(self.word2id), self.config["word_embedding_size"]],
			initializer=(tf.zeros_initializer() if self.config["emb_initial_zero"] == True else self.initializer),
			trainable=(True if self.config["train_embeddings"] == True else False))
		input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
		input_vector_size = self.config["word_embedding_size"]

		if self.config["char_embedding_size"] > 0 and self.config["char_recurrent_size"] > 0:
			with tf.variable_scope("chars"), tf.control_dependencies([tf.assert_equal(tf.shape(self.char_ids)[2], tf.reduce_max(self.word_lengths), message="Char dimensions don't match")]):
				self.char_embeddings = tf.get_variable("char_embeddings",
					shape=[len(self.char2id), self.config["char_embedding_size"]],
					initializer=self.initializer,
					trainable=True)
				char_input_tensor = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)

				s = tf.shape(char_input_tensor)
				char_input_tensor = tf.reshape(char_input_tensor, shape=[s[0]*s[1], s[2], self.config["char_embedding_size"]])
				_word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

				char_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["char_recurrent_size"],
					use_peepholes=self.config["lstm_use_peepholes"],
					state_is_tuple=True,
					initializer=self.initializer,
					reuse=False)
				char_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["char_recurrent_size"],
					use_peepholes=self.config["lstm_use_peepholes"],
					state_is_tuple=True,
					initializer=self.initializer,
					reuse=False)

				char_lstm_outputs = tf.nn.bidirectional_dynamic_rnn(char_lstm_cell_fw, char_lstm_cell_bw, char_input_tensor, sequence_length=_word_lengths, dtype=tf.float32, time_major=False)
				_, ((_, char_output_fw), (_, char_output_bw)) = char_lstm_outputs
				char_output_tensor = tf.concat([char_output_fw, char_output_bw], axis=-1)
				char_output_tensor = tf.reshape(char_output_tensor, shape=[s[0], s[1], 2 * self.config["char_recurrent_size"]])
				char_output_vector_size = 2 * self.config["char_recurrent_size"]

				if self.config["lmcost_char_gamma"] > 0.0:
					self.loss += self.config["lmcost_char_gamma"] * self.construct_lmcost(char_output_tensor, char_output_tensor, self.sentence_lengths, self.word_ids, "separate", "lmcost_char_separate")
				if self.config["lmcost_joint_char_gamma"] > 0.0:
					self.loss += self.config["lmcost_joint_char_gamma"] * self.construct_lmcost(char_output_tensor, char_output_tensor, self.sentence_lengths, self.word_ids, "joint", "lmcost_char_joint")

				if self.config["char_hidden_layer_size"] > 0:
					char_hidden_layer_size = self.config["word_embedding_size"] if self.config["char_integration_method"] == "attention" else self.config["char_hidden_layer_size"]
					char_output_tensor = tf.layers.dense(char_output_tensor, char_hidden_layer_size, activation=tf.tanh, kernel_initializer=self.initializer)
					char_output_vector_size = char_hidden_layer_size

				if self.config["char_integration_method"] == "concat":
					input_tensor = tf.concat([input_tensor, char_output_tensor], axis=-1)
					input_vector_size += char_output_vector_size
				elif self.config["char_integration_method"] == "attention":
					assert(char_output_vector_size == self.config["word_embedding_size"]), "This method requires the char representation to have the same size as word embeddings"
					static_input_tensor = tf.stop_gradient(input_tensor)
					is_unk = tf.equal(self.word_ids, self.word2id[self.UNK])
					char_output_tensor_normalised = tf.nn.l2_normalize(char_output_tensor, 2)
					static_input_tensor_normalised = tf.nn.l2_normalize(static_input_tensor, 2)
					cosine_cost = 1.0 - tf.reduce_sum(tf.multiply(char_output_tensor_normalised, static_input_tensor_normalised), axis=2)
					is_padding = tf.logical_not(tf.sequence_mask(self.sentence_lengths, maxlen=tf.shape(self.word_ids)[1]))
					cosine_cost_unk = tf.where(tf.logical_or(is_unk, is_padding), x=tf.zeros_like(cosine_cost), y=cosine_cost)
					self.loss += self.config["char_attention_cosine_cost"] * tf.reduce_sum(cosine_cost_unk)
					attention_evidence_tensor = tf.concat([input_tensor, char_output_tensor], axis=2)
					attention_output = tf.layers.dense(attention_evidence_tensor, self.config["word_embedding_size"], activation=tf.tanh, kernel_initializer=self.initializer)
					attention_output = tf.layers.dense(attention_output, self.config["word_embedding_size"], activation=tf.sigmoid, kernel_initializer=self.initializer)
					input_tensor = tf.multiply(input_tensor, attention_output) + tf.multiply(char_output_tensor, (1.0 - attention_output))
				elif self.config["char_integration_method"] == "none":
					input_tensor = input_tensor
				else:
					raise ValueError("Unknown char integration method")

		dropout_input = self.config["dropout_input"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
		input_tensor =  tf.nn.dropout(input_tensor, dropout_input, name="dropout_word")

		word_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"],
			use_peepholes=self.config["lstm_use_peepholes"],
			state_is_tuple=True,
			initializer=self.initializer,
			reuse=False)
		word_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"],
			use_peepholes=self.config["lstm_use_peepholes"],
			state_is_tuple=True,
			initializer=self.initializer,
			reuse=False)

		with tf.control_dependencies([tf.assert_equal(tf.shape(self.word_ids)[1], tf.reduce_max(self.sentence_lengths), message="Sentence dimensions don't match")]):
			(lstm_outputs_fw, lstm_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(word_lstm_cell_fw, word_lstm_cell_bw, input_tensor, sequence_length=self.sentence_lengths, dtype=tf.float32, time_major=False)

		dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
		lstm_outputs_fw =  tf.nn.dropout(lstm_outputs_fw, dropout_word_lstm)
		lstm_outputs_bw =  tf.nn.dropout(lstm_outputs_bw, dropout_word_lstm)

		if self.config["lmcost_lstm_gamma"] > 0.0:
			self.loss += self.config["lmcost_lstm_gamma"] * self.construct_lmcost(lstm_outputs_fw, lstm_outputs_bw, self.sentence_lengths, self.word_ids, "separate", "lmcost_lstm_separate")
		if self.config["lmcost_joint_lstm_gamma"] > 0.0:
			self.loss += self.config["lmcost_joint_lstm_gamma"] * self.construct_lmcost(lstm_outputs_fw, lstm_outputs_bw, self.sentence_lengths, self.word_ids, "joint", "lmcost_lstm_joint")

		processed_tensor = tf.concat([lstm_outputs_fw, lstm_outputs_bw], 2)
		processed_tensor_size = self.config["word_recurrent_size"] * 2

		if self.config["hidden_layer_size"] > 0:
			processed_tensor = tf.layers.dense(processed_tensor, self.config["hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
			processed_tensor_size = self.config["hidden_layer_size"]

		self.scores = tf.layers.dense(processed_tensor, len(self.label2id), activation=None, kernel_initializer=self.initializer, name="output_ff")

		if self.config["crf_on_top"] == True:
			crf_num_tags = self.scores.get_shape()[2].value
			self.crf_transition_params = tf.get_variable("output_crf_transitions", [crf_num_tags, crf_num_tags], initializer=self.initializer)
			log_likelihood, self.crf_transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.label_ids, self.sentence_lengths, transition_params=self.crf_transition_params)
			self.loss += self.config["main_cost"] * tf.reduce_sum(-log_likelihood)
		else:
			self.probabilities = tf.nn.softmax(self.scores)
			self.predictions = tf.argmax(self.probabilities, 2)
			loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label_ids)
			mask = tf.sequence_mask(self.sentence_lengths, maxlen=tf.shape(self.word_ids)[1])
			loss_ = tf.boolean_mask(loss_, mask)
			self.loss += self.config["main_cost"] * tf.reduce_sum(loss_)

		self.train_op = self.construct_optimizer(self.config["opt_strategy"], self.loss, self.learningrate, self.config["clip"])

	def construct_lmcost(self, input_tensor_fw, input_tensor_bw, sentence_lengths, target_ids, lmcost_type, name):
		with tf.variable_scope(name):
			lmcost_max_vocab_size = min(len(self.word2id), self.config["lmcost_max_vocab_size"])
			target_ids = tf.where(tf.greater_equal(target_ids, lmcost_max_vocab_size-1), x=(lmcost_max_vocab_size-1)+tf.zeros_like(target_ids), y=target_ids)
			cost = 0.0
			if lmcost_type == "separate":
				lmcost_fw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:]
				lmcost_bw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,:-1]
				lmcost_fw = self._construct_lmcost(input_tensor_fw[:,:-1,:], lmcost_max_vocab_size, lmcost_fw_mask, target_ids[:,1:], name=name+"_fw")
				lmcost_bw = self._construct_lmcost(input_tensor_bw[:,1:,:], lmcost_max_vocab_size, lmcost_bw_mask, target_ids[:,:-1], name=name+"_bw")
				cost += lmcost_fw + lmcost_bw
			elif lmcost_type == "joint":
				joint_input_tensor = tf.concat([input_tensor_fw[:,:-2,:], input_tensor_bw[:,2:,:]], axis=-1)
				lmcost_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:-1]
				cost += self._construct_lmcost(joint_input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids[:,1:-1], name=name+"_joint")
			else:
				raise ValueError("Unknown lmcost_type: " + str(lmcost_type))
			return cost


	def _construct_lmcost(self, input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids, name):
		with tf.variable_scope(name):
			lmcost_hidden_layer = tf.layers.dense(input_tensor, self.config["lmcost_hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
			lmcost_output = tf.layers.dense(lmcost_hidden_layer, lmcost_max_vocab_size, activation=None, kernel_initializer=self.initializer)
			lmcost_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lmcost_output, labels=target_ids)
			lmcost_loss = tf.where(lmcost_mask, lmcost_loss, tf.zeros_like(lmcost_loss))
			return tf.reduce_sum(lmcost_loss)


	def construct_optimizer(self, opt_strategy, loss, learningrate, clip):
		optimizer = None
		if opt_strategy == "adadelta":
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
		elif opt_strategy == "adam":
			optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
		elif opt_strategy == "sgd":
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
		else:
			raise ValueError("Unknown optimisation strategy: " + str(opt_strategy))

		if clip > 0.0:
			grads, vs     = zip(*optimizer.compute_gradients(loss))
			grads, gnorm  = tf.clip_by_global_norm(grads, clip)
			train_op = optimizer.apply_gradients(zip(grads, vs))
		else:
			train_op = optimizer.minimize(loss)
		return train_op


	def preload_word_embeddings(self, embedding_path):
		loaded_embeddings = set()
		embedding_matrix = self.session.run(self.word_embeddings)
		with open(embedding_path, 'r', encoding = 'utf-8') as f:
			for line in f:
				line_parts = line.strip().split()
				if len(line_parts) <= 2:
					continue
				w = line_parts[0]
				if self.config["lowercase"] == True:
					w = w.lower()
				if self.config["replace_digits"] == True:
					w = re.sub(r'\d', '0', w)
				if w in self.word2id and w not in loaded_embeddings:
					word_id = self.word2id[w]
					embedding = numpy.array(line_parts[1:])
					embedding_matrix[word_id] = embedding
					loaded_embeddings.add(w)
		self.session.run(self.word_embeddings.assign(embedding_matrix))
		print("n_preloaded_embeddings: " + str(len(loaded_embeddings)))


	def translate2id(self, token, token2id, unk_token, lowercase=False, replace_digits=False, singletons=None, singletons_prob=0.0):
		if lowercase == True:
			token = token.lower()
		if replace_digits == True:
			token = re.sub(r'\d', '0', token)

		token_id = None
		if singletons != None and token in singletons and token in token2id and unk_token != None and numpy.random.uniform() < singletons_prob:
			token_id = token2id[unk_token]
		elif token in token2id:
			token_id = token2id[token]
		elif unk_token != None:
			token_id = token2id[unk_token]
		else:
			raise ValueError("Unable to handle value, no UNK token: " + str(token))
		return token_id


	def create_input_dictionary_for_batch(self, batch, is_training, learningrate):
		sentence_lengths = numpy.array([len(sentence) for sentence in batch])
		max_sentence_length = sentence_lengths.max()
		# print(batch[0])
		# outer = []
		# for sentence in batch : 
		# 	l = []
		# 	if len(sentence) == 0 : 
		# 		print("true")
		# 	for word in sentence:
		# 		l.append(len(word[0]))
		# 	m = numpy.array(l)
		# 	maximum = m.max()
		# 	outer.append(maximum)
		# outer = numpy.array(outer).max()



		max_word_length = numpy.array([numpy.array([len(word[0]) for word in sentence]).max() for sentence in batch]).max()
		if self.config["allowed_word_length"] > 0 and self.config["allowed_word_length"] < max_word_length:
			max_word_length = min(max_word_length, self.config["allowed_word_length"])

		word_ids = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
		char_ids = numpy.zeros((len(batch), max_sentence_length, max_word_length), dtype=numpy.int32)
		word_lengths = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
		label_ids = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)

		singletons = self.singletons if is_training == True else None
		singletons_prob = self.config["singletons_prob"] if is_training == True else 0.0
		for i in range(len(batch)):
			for j in range(len(batch[i])):
				word_ids[i][j] = self.translate2id(batch[i][j][0], self.word2id, self.UNK, lowercase=self.config["lowercase"], replace_digits=self.config["replace_digits"], singletons=singletons, singletons_prob=singletons_prob)
				label_ids[i][j] = self.translate2id(batch[i][j][3], self.label2id, None)
				word_lengths[i][j] = min(len(batch[i][j][0]), max_word_length)
				for k in range(min(len(batch[i][j][0]), max_word_length)):
					char_ids[i][j][k] = self.translate2id(batch[i][j][0][k], self.char2id, self.CUNK)

		input_dictionary = {self.word_ids: word_ids, self.char_ids: char_ids, self.sentence_lengths: sentence_lengths, self.word_lengths: word_lengths, self.label_ids: label_ids, self.learningrate: learningrate, self.is_training: is_training}
		return input_dictionary


	def viterbi_decode(self, score, transition_params):
		trellis = numpy.zeros_like(score)
		backpointers = numpy.zeros_like(score, dtype=numpy.int32)
		trellis[0] = score[0]

		for t in range(1, score.shape[0]):
			v = numpy.expand_dims(trellis[t - 1], 1) + transition_params
			trellis[t] = score[t] + numpy.max(v, 0)
			backpointers[t] = numpy.argmax(v, 0)

		viterbi = [numpy.argmax(trellis[-1])]
		for bp in reversed(backpointers[1:]):
			viterbi.append(bp[viterbi[-1]])
		viterbi.reverse()

		viterbi_score = numpy.max(trellis[-1])
		return viterbi, viterbi_score, trellis


	def process_batch(self, batch, is_training, learningrate):
		feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learningrate)
		# print ("@"*100)
		# print(batch[0])
		# print(feed_dict.keys())
		final_result = []
		if self.config["crf_on_top"] == True:
			cost, scores = self.session.run([self.loss, self.scores] + ([self.train_op] if is_training == True else []), feed_dict=feed_dict)[:2]
			predicted_labels = []
			predicted_probs = []


			for i in range(len(batch)):
				result_formatted =  list()
				sentence_length = len(batch[i])
				viterbi_seq, viterbi_score, viterbi_trellis = self.viterbi_decode(scores[i], self.session.run(self.crf_transition_params))
				predicted_labels.append(viterbi_seq[:sentence_length])
				predicted_probs.append(viterbi_trellis[:sentence_length])
				for ind, item in enumerate(batch[i]):
					temp = item
					temp.append(self.id2label[viterbi_trellis[:sentence_length]] )
					result_formatted.append(temp)
				final_result.append(result_formatted)

		else:
			cost, predicted_labels_, predicted_probs_ = self.session.run([self.loss, self.predictions, self.probabilities] + ([self.train_op] if is_training == True else []), feed_dict=feed_dict)[:3]
			predicted_labels = []
			predicted_probs = []

			for i in range(0,len(batch)):
				result_formatted =  list()
				sentence_length = len(batch[i])
				predicted_labels.append(predicted_labels_[i][:sentence_length])
				predicted_probs.append(predicted_probs_[i][:sentence_length])
				# print(batch[1], predicted_labels[0])

				# print(batch[i])

				for ind, item in enumerate(batch[i]):
					temp = [None,None]
					temp.append(item[0])
					temp.append(self.id2label[predicted_labels_[i][:sentence_length][ind]] )
					result_formatted.append(temp)
				final_result.append(result_formatted)


		return cost, predicted_labels, predicted_probs, final_result


	def initialize_session(self):
		tf.set_random_seed(self.config["random_seed"])
		session_config = tf.ConfigProto()
		session_config.gpu_options.allow_growth = self.config["tf_allow_growth"]
		session_config.gpu_options.per_process_gpu_memory_fraction = self.config["tf_per_process_gpu_memory_fraction"]
		self.session = tf.Session(config=session_config)
		self.session.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=1)


	def get_parameter_count(self):
		total_parameters = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape()
			variable_parameters = 1
			for dim in shape:
				variable_parameters *= dim.value
			total_parameters += variable_parameters
		return total_parameters


	def get_parameter_count_without_word_embeddings(self):
		shape = self.word_embeddings.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		return self.get_parameter_count() - variable_parameters


	def save_metadata(self, filename):
		# print ("inside save")
		dump = {}
		dump["config"] = self.config
		dump["UNK"] = self.UNK
		dump["CUNK"] = self.CUNK
		dump["word2id"] = self.word2id
		dump["char2id"] = self.char2id
		dump["label2id"] = self.label2id
		dump["singletons"] = self.singletons
		dump["id2label"] = self.id2label

		dump["params"] = {}
		for variable in tf.global_variables():
			assert(variable.name not in dump["params"]), "Error: variable with this name already exists" + str(variable.name)
			dump["params"][variable.name] = self.session.run(variable)
		# print(filename)
		with open(filename, 'wb') as f:
			pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)


	@staticmethod
	def load_metadata(filename):
		print("loading the model...\n\n")
		print(filename)
		with open(filename, 'rb') as f:
			dump = pickle.load(f)

			# for safety, so we don't overwrite old models
			dump["config"]["save"] = None

			labeler = Sequence_labeler(dump["config"])
			labeler.UNK = dump["UNK"]
			labeler.CUNK = dump["CUNK"]
			labeler.word2id = dump["word2id"]
			labeler.char2id = dump["char2id"]
			labeler.label2id = dump["label2id"]
			labeler.singletons = dump["singletons"]
			labeler.id2label = dump["id2label"]

			labeler.construct_network()
			labeler.initialize_session()
			labeler.load_params(filename)

			return labeler


	def load_params(self, filename):
		with open(filename, 'rb') as f:
			dump = pickle.load(f)

			for variable in tf.global_variables():
				assert(variable.name in dump["params"]), "Variable not in dump: " + str(variable.name)
				assert(variable.shape == dump["params"][variable.name].shape), "Variable shape not as expected: " + str(variable.name) + " " + str(variable.shape) + " " + str(dump["params"][variable.name].shape)
				value = numpy.asarray(dump["params"][variable.name])
				self.session.run(variable.assign(value))
	def convert_ground_truth(self, data, *args, **kwargs):

		pass

	def train(self, data,  data_dev, temp_model_path):
		data_train = data

		if data_train != None:
			model_selector = self.config["model_selector"].split(":")[0]
			model_selector_type = self.config["model_selector"].split(":")[1]
			best_selector_value = 0.0
			best_epoch = -1
			learningrate = self.config["learningrate"]
			for epoch in range(self.config["epochs"]):
				print("EPOCH: " + str(epoch))
				print("current_learningrate: " + str(learningrate))
				random.shuffle(data_train)

				results_train,a,b, c = self.process_sentences(data_train, is_training=True, learningrate=learningrate, config=self.config, name="train")

				if data_dev != None:
					total_cost_Dev, total_predicted_labels_dev, total_predicted_probs_dev, predictions_formatted = self.process_sentences(data_dev, is_training=False, learningrate=0.0, config=self.config, name="dev")
					precision,recall,f1 = self.evaluate(total_predicted_labels_dev,data_dev, total_cost_Dev,"dev")
					# for key in results_dev:
					# 	print(key + ": " + str(results_dev[key]))
					# if math.isnan(results_dev["dev_cost_sum"]) or math.isinf(results_dev["dev_cost_sum"]):
					# 	sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
					# 	break
					print("precision: ",precision," recall: ",recall," F1 : ",f1)
					if (epoch == 0 or (model_selector_type == "high" and f1 > best_selector_value)
								   or (model_selector_type == "low" and f1 < best_selector_value)):
						best_epoch = epoch
						best_selector_value = f1
						# print("saving ",temp_model_path)
						self.save_model(temp_model_path)
						self.save_metadata('model_pickle.p')

					print("best_epoch: " + str(best_epoch))

					if self.config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= self.config["stop_if_no_improvement_for_epochs"]:
						break

					if (epoch - best_epoch) > 3:
						learningrate *= self.config["learningrate_decay"]

				while self.config["garbage_collection"] == True and gc.collect() > 0:
					pass

	def save_model(self,temp_model_path):
		self.saver.save(self.session, temp_model_path, latest_filename=os.path.basename(temp_model_path)+".checkpoint")

	def load_model(self,sess,temp_model_path):
		# saver.restore(sess, "/tmp/model.ckpt")
		self.saver.restore(self.session, temp_model_path)

	def predict(self, data_test ):
		results_test = []
		if self.config["path_test"] is not None:
			total_cost, total_predicted_labels , total_predicted_probs, predictions_formatted = self.process_sentences(data_test, is_training=False, learningrate=0.0, config= self.config, name="test")

		return predictions_formatted
	# 	return total_cost, total_predicted_labels , total_predicted_probs


	def evaluate(self, predictions, groundTruths,cost,name):

		evaluator = SequenceLabelingEvaluator(self.config["main_label"], self.label2id, self.config["conll_eval"])
		evaluator.append_data(cost, groundTruths, predictions)

		# self.word_ids, self.char_ids, self.char_mask, self.label_ids = None, None, None, None
		while self.config["garbage_collection"] == True and gc.collect() > 0:
			pass

		results = evaluator.get_results(name)

		return results[name+"_p"],results[name+"_r"],results[name+"_f"]


	def process_sentences(self, data, is_training, learningrate, config, name):
		"""
		Process all the sentences with the labeler, return evaluation metrics.
		"""


		total_cost, total_predicted_labels, total_predicted_probs, total_predictions_formatted = 0.0,[],[],[]

		batches_of_sentence_ids = self.create_batches_of_sentence_ids(data, self.config["batch_equal_size"], self.config["max_batch_size"])
		if is_training == True:
			random.shuffle(batches_of_sentence_ids)

		for sentence_ids_in_batch in batches_of_sentence_ids:
			batch = [data[i] for i in sentence_ids_in_batch]
			cost, predicted_labels, predicted_probs, result_formatted= self.process_batch(batch, is_training, learningrate)
			total_cost += cost
			total_predicted_probs.extend(predicted_probs)
			total_predicted_labels.extend(predicted_labels)
			total_predictions_formatted.extend(result_formatted)

		
		return total_cost,total_predicted_labels, total_predicted_probs, total_predictions_formatted






def main(input_file = ""):
	path = "conf/fcepublic.conf"
	config = parse_config("config", path)
	# temp_model_path =  path + ".model"
	temp_model_path = "model/model"
	labeler = Sequence_labeler(config)
	if input_file == "":
		file_paths = { "train": config["path_train"] , "dev" : config["path_dev"], "test" : config["path_test"]}
	else:
		file_paths =  { "train": input_file , "dev" :input_file , "test" : input_file}
	dataset = labeler.read_dataset(file_paths,"ConLL")

	dataset["train"] = labeler.data_formatted(dataset["train"])
	dataset["dev"] = labeler.data_formatted(dataset["dev"])
	dataset["test"] = labeler.data_formatted(dataset["test"])
	print("after data_formatted")
	print("train",len(dataset["train"]))
	print("dev",len(dataset["dev"]))
	print("test",len(dataset["test"]))
	data_train, data_dev, data_test = dataset["train"],dataset["dev"],dataset["test"]
	

	labeler.build_vocabs(data_train, data_dev, data_test, config["preload_vectors"])
	labeler.construct_network()
	labeler.initialize_session()
	if config["preload_vectors"] != None:
		labeler.preload_word_embeddings(config["preload_vectors"])
	if config["load"] != None and len(config["load"])  > 0 and os.path.exists(config["load"]):
		try:
			labeler.load_model(labeler.session, temp_model_path)
		except:
			print("error in loading the model")
	
	print("parameter_count: " + str(labeler.get_parameter_count()))
	print("parameter_count_without_word_embeddings: " + str(labeler.get_parameter_count_without_word_embeddings()))
	labeler.train(data_train,data_dev,temp_model_path)
	predictions_formatted = labeler.predict(data_test)
	print("predict done")
	
	#print(predictions_formatted)

	cost = 0
	predicted_labels = []
	groundTruths = []

	for i in range(len(predictions_formatted)):
		predicted_labels_sent = []
		groundTruths_sent = []
		for j in range(len(predictions_formatted[i])):
			predicted_labels_sent.append(labeler.label2id[predictions_formatted[i][j][-1]])
			
		predicted_labels.append(predicted_labels_sent)
		

	for i in data_test:
		groundTruths_sent = []
		for item in i : 
			groundTruths_sent.append(item)
		groundTruths.append(groundTruths_sent)

	with open('result.txt','w',encoding = 'utf-8') as f:
		for i in range(len(data_test)):
			for j in range(len(data_test[i])):

				
				f.write(str(data_test[i][j][0])+" "+str(data_test[i][j][3])+ " " +  str(labeler.id2label[predicted_labels[i][j]]))
				f.write("\n")
			if ( i != len(data_test)-1):
				f.write("\n")



	precision,recall,f1 = labeler.evaluate(predicted_labels,groundTruths, cost,"test")
	print(precision,recall,f1)

	return "result.txt"


if __name__ == '__main__':
	main()
