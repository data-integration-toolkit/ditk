import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from ner import Ner
from utils import parseconfig
from utils.conll2003_prepro import process_data
from utils import batchnize_dataset
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from models import BaseModel, AttentionCell, multi_head_attention
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from models.nns import multi_conv1d, highway_network, layer_normalize
from utils import Progbar
from tensorflow.contrib.crf import viterbi_decode, crf_log_likelihood
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, MultiRNNCell
from utils import CoNLLeval, load_dataset, get_logger, process_batch_data, align_data
from utils.common import word_convert, UNK




class NeuralSequenceLabeler(Ner,BaseModel):
	def __init__(self,config):
		self.config = config
		super(NeuralSequenceLabeler, self).__init__(config)

	def convert_ground_truth(self, data, *args, **kwargs):
		pass

	def read_dataset(self,file_dict, dataset_name=None, *args, **kwargs):
	
		standard_split = ["train", "test", "dev"]
		data = {}
		try:
			for split in standard_split:
				file = file_dict[split]
				print ("split",file)
				with open(file, mode='r', encoding='utf-8') as f:
					raw_data = f.read().splitlines()
				for i, line in enumerate(raw_data):
					if len(line.strip()) > 0:
						raw_data[i] = line.strip().split()
					else:
						raw_data[i] = list(line.strip())
				data[split] = raw_data
		except KeyError:
			raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
		except Exception as e:
			print('Something went wrong.', e)
		print("inside read_dataset")
		print("train",len(data["train"]))
		print("dev",len(data["dev"]))
		print("test",len(data["test"]))
		return data


	# def read_dataset(self, file_dict, dataset_name,max_sentence_length=-1):
	# 	dataset = dict()
	# 	for dataset_type,file_paths in file_dict.items():
	# 		sentences = []
	# 		line_length = None
	# 		for file_path in file_paths.strip().split(","):
	# 			with open(file_path, "r") as f:
	# 				sentence = []
	# 				for line in f:
	# 					line = line.strip()
	# 					if len(line) > 0:
	# 						line_parts = line.split()


	# 						assert(len(line_parts) >= 2)
	# 						assert(len(line_parts) == line_length or line_length == None)
	# 						line_length = len(line_parts)
	# 						sentence.append(line_parts)
	# 					elif len(line) == 0 and len(sentence) > 0:
	# 						if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
	# 							sentences.append(sentence)
	# 						sentence = []
	# 				if len(sentence) > 0:
	# 					if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
	# 						sentences.append(sentence)
	# 		dataset[dataset_type] = sentences
	# 	return dataset

	def read_dataset_helper(self, file_dict, dataset_name):
		dataset = dict()
		print(len(file_dict['train']))
		print(len(file_dict['dev']))
		print(len(file_dict['test']))
		train_set = batchnize_dataset(file_dict['train'], self.config["batch_size"], shuffle=True)
		# used for computing validate loss
		valid_data = batchnize_dataset(file_dict['dev'], batch_size=1000, shuffle=True)[0]
		# used for computing validate accuracy, precision, recall and F1 scores
		valid_set = batchnize_dataset(file_dict['dev'], self.config["batch_size"], shuffle=False)
		# used for computing test accuracy, precision, recall and F1 scores
		test_set = batchnize_dataset(file_dict['test'], self.config["batch_size"], shuffle=False)
		dataset["train_set"] = train_set
		dataset["dev_data"] = valid_data
		dataset["dev_set"] = valid_set
		dataset["test_set"] = test_set

		return dataset

	def _add_placeholders(self):
		self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
		self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")
		self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
		if self.cfg["use_chars"]:
			self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
			self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
		# hyper-parameters
		self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
		self.batch_size = tf.placeholder(tf.int32, name="batch_size")
		self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
		self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
		self.lr = tf.placeholder(tf.float32, name="learning_rate")

	def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
		feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
		if "tags" in batch:
			feed_dict[self.tags] = batch["tags"]
		if self.cfg["use_chars"]:
			feed_dict[self.chars] = batch["chars"]
			feed_dict[self.char_seq_len] = batch["char_seq_len"]
		feed_dict[self.keep_prob] = keep_prob
		feed_dict[self.drop_rate] = 1.0 - keep_prob
		feed_dict[self.is_train] = is_train
		if lr is not None:
			feed_dict[self.lr] = lr
		return feed_dict

	def _create_rnn_cell(self):
		if self.cfg["num_layers"] is None or self.cfg["num_layers"] <= 1:
			return self._create_single_rnn_cell(self.cfg["num_units"])
		else:
			if self.cfg["use_stack_rnn"]:
				return [self._create_single_rnn_cell(self.cfg["num_units"]) for _ in range(self.cfg["num_layers"])]
			else:
				return MultiRNNCell([self._create_single_rnn_cell(self.cfg["num_units"])
									 for _ in range(self.cfg["num_layers"])])

	def _build_embedding_op(self):
		with tf.variable_scope("embeddings"):
			if not self.cfg["use_pretrained"]:
				self.word_embeddings = tf.get_variable(name="emb", dtype=tf.float32, trainable=True,
													   shape=[self.word_vocab_size, self.cfg["emb_dim"]])
			else:
				self.word_embeddings = tf.Variable(np.load(self.cfg["pretrained_emb"])["embeddings"], name="emb",
												   dtype=tf.float32, trainable=self.cfg["tuning_emb"])
			word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
			print("word embedding shape: {}".format(word_emb.get_shape().as_list()))
			if self.cfg["use_chars"]:
				self.char_embeddings = tf.get_variable(name="c_emb", dtype=tf.float32, trainable=True,
													   shape=[self.char_vocab_size, self.cfg["char_emb_dim"]])
				char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars, name="chars_emb")
				char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"], self.cfg["channel_sizes"],
											  drop_rate=self.drop_rate, is_train=self.is_train)
				print("chars representation shape: {}".format(char_represent.get_shape().as_list()))
				word_emb = tf.concat([word_emb, char_represent], axis=-1)
			if self.cfg["use_highway"]:
				self.word_emb = highway_network(word_emb, self.cfg["highway_layers"], use_bias=True, bias_init=0.0,
												keep_prob=self.keep_prob, is_train=self.is_train)
			else:
				self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
			print("word and chars concatenation shape: {}".format(self.word_emb.get_shape().as_list()))

	def _build_model_op(self):
		with tf.variable_scope("bi_directional_rnn"):
			cell_fw = self._create_rnn_cell()
			cell_bw = self._create_rnn_cell()
			if self.cfg["use_stack_rnn"]:
				rnn_outs, *_ = stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_emb, dtype=tf.float32,
															   sequence_length=self.seq_len)
			else:
				rnn_outs, *_ = bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_emb, sequence_length=self.seq_len,
														 dtype=tf.float32)
			rnn_outs = tf.concat(rnn_outs, axis=-1)
			rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate, training=self.is_train)
			if self.cfg["use_residual"]:
				word_project = tf.layers.dense(self.word_emb, units=2 * self.cfg["num_units"], use_bias=False)
				rnn_outs = rnn_outs + word_project
			outputs = layer_normalize(rnn_outs) if self.cfg["use_layer_norm"] else rnn_outs
			print("rnn output shape: {}".format(outputs.get_shape().as_list()))

		if self.cfg["use_attention"] == "self_attention":
			with tf.variable_scope("self_attention"):
				attn_outs = multi_head_attention(outputs, outputs, self.cfg["num_heads"], self.cfg["attention_size"],
												 drop_rate=self.drop_rate, is_train=self.is_train)
				if self.cfg["use_residual"]:
					attn_outs = attn_outs + outputs
				outputs = layer_normalize(attn_outs) if self.cfg["use_layer_norm"] else attn_outs
				print("self-attention output shape: {}".format(outputs.get_shape().as_list()))

		elif self.cfg["use_attention"] == "normal_attention":
			with tf.variable_scope("normal_attention"):
				context = tf.transpose(outputs, [1, 0, 2])
				p_context = tf.layers.dense(outputs, units=2 * self.cfg["num_units"], use_bias=False)
				p_context = tf.transpose(p_context, [1, 0, 2])
				attn_cell = AttentionCell(self.cfg["num_units"], context, p_context)  # time major based
				attn_outs, _ = dynamic_rnn(attn_cell, context, sequence_length=self.seq_len, time_major=True,
										   dtype=tf.float32)
				outputs = tf.transpose(attn_outs, [1, 0, 2])
				print("attention output shape: {}".format(outputs.get_shape().as_list()))

		with tf.variable_scope("project"):
			self.logits = tf.layers.dense(outputs, units=self.tag_vocab_size, use_bias=True)
			print("logits shape: {}".format(self.logits.get_shape().as_list()))

	def train_epoch(self, train_set, valid_data, epoch):	
		num_batches = len(train_set)
		prog = Progbar(target=num_batches)
		for i, batch_data in enumerate(train_set):
			feed_dict = self._get_feed_dict(batch_data, is_train=True, keep_prob=self.cfg["keep_prob"],
											lr=self.cfg["lr"])
			_, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
			cur_step = (epoch - 1) * num_batches + (i + 1)
			prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
			self.train_writer.add_summary(summary, cur_step)
			if i % 100 == 0:
				valid_feed_dict = self._get_feed_dict(valid_data)
				valid_summary = self.sess.run(self.summary, feed_dict=valid_feed_dict)
				self.test_writer.add_summary(valid_summary, cur_step)

	def train(self, train_set, valid_data, valid_set, test_set):
		self.logger.info("Start training...")
		best_f1, no_imprv_epoch, init_lr = -np.inf, 0, self.cfg["lr"]
		self._add_summary()
		for epoch in range(1, self.cfg["epochs"] + 1):
			self.logger.info('Epoch {}/{}:'.format(epoch, self.cfg["epochs"]))
			self.train_epoch(train_set, valid_data, epoch)  # train epochs
			if self.cfg["use_lr_decay"]:  # learning rate decay
				self.cfg["lr"] = max(init_lr / (1.0 + self.cfg["lr_decay"] * epoch), self.cfg["minimal_lr"])

			# self.predict_helper(valid_set, "dev")
			predictions, groundtruth,words_list, save_path,name = self.predict_helper(test_set, "dev")
			score = self.evaluate(predictions, groundtruth,words_list, save_path,name)
			cur_test_score = score["FB1"]
			if cur_test_score > best_f1:
				best_f1 = cur_test_score
				no_imprv_epoch = 0
				if epoch%self.cfg['store_checkpoint']==0:
					self.save_model(epoch)
				self.logger.info(' -- new BEST score on test dataset: {:04.2f}'.format(best_f1))
			else:
				no_imprv_epoch += 1
				if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
					self.logger.info('early stop at {}th epoch without improvement, BEST score on testset: {:04.2f}'
									 .format(epoch, best_f1))
					break
		self.train_writer.close()
		self.test_writer.close()

	def evaluate(self,predictions, groundTruths,words_list, save_path,name):
		ce = CoNLLeval()
		score = ce.conlleval(predictions, groundTruths, words_list, save_path)
		# print(score)
		self.logger.info("{} dataset -- pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}".format(name, score["precision"], score["recall"], score["FB1"]))
		
		return score



	def load_model(self, ckpt_path=None):
		print("in load model....\n\n")
		if ckpt_path is not None:
			ckpt = tf.train.get_checkpoint_state(ckpt_path)
		else:
			ckpt = tf.train.get_checkpoint_state(self.cfg["checkpoint_path"])  # get checkpoint state
		if ckpt and ckpt.model_checkpoint_path:  # restore session
			print("inside if.. loaded session")
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)

	def save_model(self, epoch):
		self.saver.save(self.sess, self.cfg["checkpoint_path"] + self.cfg["model_name"], global_step=epoch)


	def predict_helper(self, dataset, name):
		print("\n\nin predict...")
		save_path = os.path.join(self.cfg["checkpoint_path"], "result.txt")
		predictions, groundtruth, words_list = list(), list(), list()
		for data in dataset:
			predicts = self._predict_op(data)
			# print("type of predicts is ",type(predicts))
			for tags, preds, words, seq_len in zip(data["tags"], predicts, data["words"], data["seq_len"]):
				tags = [self.rev_tag_dict[x] for x in tags[:seq_len]]
				preds = [self.rev_tag_dict[x] for x in preds[:seq_len]]
				words = [self.rev_word_dict[x] for x in words[:seq_len]]
				predictions.append(preds)
				groundtruth.append(tags)
				words_list.append(words)
				# print("@"*100)
				# print(preds)
				# print(tags)
				# print(words)
		# print("groundtruth len",len(groundtruth)," ", predictions[0]," ",groundtruth[0])


		return predictions, groundtruth,words_list, save_path,name



	def predict(self, dataset, name):
		print("\n\nin predict...")
		save_path = os.path.join(self.cfg["checkpoint_path"], "result.txt")
		predictions, groundtruth, words_list = list(), list(), list()
		for data in dataset:
			predicts = self._predict_op(data)
			# print("type of predicts is ",type(predicts))
			for tags, preds, words, seq_len in zip(data["tags"], predicts, data["words"], data["seq_len"]):
				tags = [self.rev_tag_dict[x] for x in tags[:seq_len]]
				preds = [self.rev_tag_dict[x] for x in preds[:seq_len]]
				words = [self.rev_word_dict[x] for x in words[:seq_len]]
				predictions.append(preds)
				groundtruth.append(tags)
				words_list.append(words)
				# print("@"*100)
				# print(preds)
				# print(tags)
				# print(words)
		# print("groundtruth len",len(groundtruth)," ", predictions[0]," ",groundtruth[0])
		total_results = []
		for i in range(len(groundtruth)):
			result = []
			for j in range(len(groundtruth[i])):
				result.append([None, None, words_list[i][j],predictions[i][j]])
			total_results.append(result)

		# print(total_results)


		return total_results



	def _predict_op(self, data):
		feed_dict = self._get_feed_dict(data)
		if self.cfg["use_crf"]:
			logits, trans_params, seq_len = self.sess.run([self.logits, self.trans_params, self.seq_len],feed_dict=feed_dict)
			return self.viterbi_decode(logits, trans_params, seq_len)
		else:
			pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
			logits = self.sess.run(pred_logits, feed_dict=feed_dict)
			return logits



def main(input_file = ""):

	config = parseconfig.parseConfig()
	if input_file == "":
		file_dict = {"train":os.path.join(config["raw_path"], "train1.txt"),"dev":os.path.join(config["raw_path"], "valid1.txt"),"test":os.path.join(config["raw_path"], "test1.txt")}
	else:
		file_dict =  { "train": input_file , "dev" :input_file , "test" : input_file}
	# file_dict = {"train":os.path.join(config["raw_path"], "train1.txt"),"dev":os.path.join(config["raw_path"], "valid1.txt"),"test":os.path.join(config["raw_path"], "test1.txt")}
	neuralSequenceLabeler =  NeuralSequenceLabeler(config)
	dataset = neuralSequenceLabeler.read_dataset(file_dict,"conll2003")
	# print(dataset["train"][:10])
	train_set,dev_set,test_set,vocab = process_data(dataset,config)
	neuralSequenceLabeler.initialize_metadata(vocab)




	file_dict = {"train": train_set, "dev":dev_set,"test":test_set }
	# file_dict = {"train": train_set[:100], "dev":dev_set[:20],"test":test_set[:100] }
	try:
		neuralSequenceLabeler.load_model()
	except:
		print("Error loading the model")
		pass

	dataset = neuralSequenceLabeler.read_dataset_helper(file_dict,"conll2012")
	neuralSequenceLabeler.train(dataset["train_set"],dataset["dev_data"],dataset["dev_set"],dataset["test_set"])
	# predictions = neuralSequenceLabeler.predict(dataset["test_set"])

	# print(predictions[0])
	# predictions, groundtruth,words_list, save_path,name = neuralSequenceLabeler.predict_helper(dataset["test_set"],"test")
	#
	#
	# print(predictions[10])
	#
	# print("\n\n#####")
	# print(groundtruth[10])
	predictions_formatted = neuralSequenceLabeler.predict(dataset["test_set"],"test")
	# print("main results")
	# 

	predictions = []
	groundTruths = []
	words_list = []
	for i in range(len(predictions_formatted)):
		predictions_sentence = []
		groundTruths_sentence = []
		words_list_sentence = []
		for j in range(len(predictions_formatted[i])):
			# print(dataset["test_set"][i]["tags"])
			# print(len(dataset["test_set"][i]["tags"])," length of tags  ")
			# print("j is : ",j)
			words_list_sentence.append(predictions_formatted[i][j][2])
			# groundTruths_sentence.append(dataset["test_set"][i]["tags"][j])
			# groundTruths_sentence.append(predictions_formatted[i][j][1])
			predictions_sentence.append(predictions_formatted[i][j][3])

		predictions.append(predictions_sentence)
		# groundTruths.append(groundTruths_sentence)
		words_list.append(words_list_sentence)
	for data in dataset["test_set"]:
		for tags,  seq_len in zip(data["tags"], data["seq_len"]):
				tags = [neuralSequenceLabeler.rev_tag_dict[x] for x in tags[:seq_len]]
				groundTruths.append(tags)
	print("groundTruths....")
	print(groundTruths[:4])
	save_path = os.path.join(neuralSequenceLabeler.cfg["checkpoint_path"], "result.txt")
	name = "test"
	score = neuralSequenceLabeler.evaluate(predictions, groundTruths,words_list, save_path,name)

	print("Test Data")
	neuralSequenceLabeler.logger.info("{} dataset -- pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}".format(name, score["precision"], score["recall"], score["FB1"]))
	
	return os.path.join(neuralSequenceLabeler.cfg["checkpoint_path"], "result.txt")





if __name__ == '__main__':
	main()
