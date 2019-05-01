
#Python 3.x

from abc import ABCMeta, abstractmethod
from relation_extraction_3 import RelationExtraction
import util
import os
import numpy as np
import train as cnn_train
import tensorflow as tf
from datetime import datetime

#import eval

#THIS_DIR = os.path.abspath(os.path.dirname(__file__))
#data_dir = os.path.join(THIS_DIR, 'data')

class MLMI_CNN_Model(RelationExtraction):
	def __init__(self):
		pass

	def read_dataset(self, input_file, *args, **kwargs): 
		
		rel = set()
		entity = set()
		df=[]
		with open(input_file) as fin:
			for line in fin:
				x = line.strip().split('\t')
				rel.add(x[9])
				e1 = x[1].strip().lower().replace(' ','')
				e2 = x[5].strip().lower().replace(' ','')
				entity.add(e1)
				entity.add(e2)
				df.append(x)


		rel_id={}
		for i,r in enumerate(rel):
			rel_id[r] = '<P'+ str(i + 10) + '>'

		entity_id = {}
		for i,e in enumerate(entity):
			entity_id[e] = '<Q'+ str(i + 10) + '>'
		
		rel_all_P = []
		rel_all_text = []
		for k,v in rel_id.items():
			rel_all_P.append(v)
			rel_all_text.append(k)
		self.num_classes = len(rel_all_P)
		self.relationtypes = rel_all_text


		final_sentence=[]
		final_target = []

		for row in df:
			e1 = row[1].strip().lower().replace(' ','')
			e2 = row[5].strip().lower().replace(' ','')
			rel = row[9]
			e1id = entity_id[e1]
			e2id = entity_id[e2]
			relid = rel_id[rel]
			if not str(row[3]).isdigit():
				continue
			if not str(row[4]).isdigit():
				continue
			if not str(row[7]).isdigit():
				continue
			if not str(row[8]).isdigit():
				continue
			e1_b = int(row[3])
			e1_e = int(row[4])
			e2_b = int(row[7])
			e2_e = int(row[8])
			sen = row[0]
			new_sen = sen[:e1_b] + e1id + sen[e1_e:e2_b] + e2id + sen[e2_e:]
			target = [0]*len(rel_all_P)
			idx = rel_all_P.index(relid)
			target[idx] = 1
			final_sentence.append(new_sen)
			final_target.append(target)

		model_path = 'mlmi_datasets'
		if not os.path.exists(model_path):
			os.mkdir(model_path)
		with open(model_path + '/source.txt','w') as f1:
			for i in final_sentence:
				f1.write(i + '\n')
		with open(model_path + '/target.txt','w') as f2:
			for i in final_target:
				i = [str(x) for x in i]
				t = ' '.join(i)
				f2.write(t + '\n')

		with open(model_path + '/entity_id.txt','w') as f3:
			for k,v in entity_id.items():
				f3.write(k + '\t' + v + '\n')

		with open(model_path + '/relation_id.txt','w') as f3:
			for k,v in rel_id.items():
				f3.write(k + '\t' + v + '\n')

		return model_path



	def data_preprocess(self,input_data, *args, **kwargs):
		vocab_path = input_data + '/vocab.txt'
		data_path = input_data + '/source.txt'
		target_path = input_data + '/ids.txt'
		max_vocab_size = 36500
		util.create_vocabulary(vocab_path, data_path, max_vocab_size)
		#util.prepare_ids(input_file , vocab_path)
		util.data_to_token_ids(data_path, target_path, vocab_path, bos=True, eos=True)

		max_len = 0
		with open(target_path) as fid:
			for line in fid:
				x = line.split()
				max_len = max(len(x),max_len)

		with open(vocab_path) as ff:
			lens = 0
			for i, l in enumerate(ff):
				lens +=1
			self.vocab_size = lens+1
		self.sent_len = max_len + 2


		#pretrained embeddings
		embedding_path = 'word2vec/GoogleNews-vectors-negative300.bin'
		if os.path.exists(embedding_path):
			word2id, _ = util.initialize_vocabulary(vocab_path)
			embedding = util.prepare_pretrained_embedding(embedding_path, word2id)
			np.save(input_data + '/emb.npy', embedding)
		else:
			print("Pretrained embeddings file %s not found." % embedding_path)

	def tokenize(self, input_data ,ngram_size=None, *args, **kwargs):  
		pass


	def train(self, train_data, *args, **kwargs): 

		if not os.path.exists('train'):
			os.mkdir('train')
		id_path = train_data + '/ids.txt'
		target_path = train_data + '/target.txt'
		atten_path = train_data + '/source.att'
		train_data, test_data =util.read_data(id_path,target_path,self.sent_len, train_size = 100000,shuffle=False)

 
		results_path = cnn_train.train(train_data, test_data,self.sent_len,self.vocab_size,self.num_classes,multi_label = True, use_pretrain = True)
		return results_path



	def evaluate(self, input_data, trained_model = None, *args, **kwargs):
		pre, rec = zip(*self.evals)

		auc = util.calc_auc_pr(pre, rec)
		f1 = (2.0 * pre[5] * rec[5]) / (pre[5] + rec[5])
		print('%s:  p = %.4f, r = %4.4f, f1 = %.4f' % (datetime.now(), pre[5], rec[5], f1))
		return [pre[5],rec[5],[f1]]


	def predict(self, test_data, entity_1 = None, entity_2= None,  trained_model = None, *args, **kwargs):   

		vocab_path = test_data + '/vocab.txt'
		source_path = test_data + '/source.txt'
		target_path = test_data + '/target.txt'
		#_,vocab = util.initialize_vocabulary(vocab_path)
		#vocab = dict([(x, y) for (x, y) in enumerate(vocab)])

		#print(vocab)
		test_path = test_data

		_,test_data = self.get_model_data( test_data + '/ids.txt',test_data + '/target.txt')


		config = util.load_from_dump(trained_model + '/flags.cPickle')
		config['train_dir'] = trained_model
		with tf.Graph().as_default():
			with tf.variable_scope('cnn'):
				import cnn
				m = cnn.Model(config, is_train=False)
				saver = tf.train.Saver(tf.global_variables())

			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(config['train_dir'])
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
				else:
					raise IOError("Loading checkpoint file failed!")

				print("\nStart evaluation on test set ...\n")
				x_batch, y_batch, _ = zip(*test_data)
				feed = {m.inputs: np.array(x_batch), m.labels: np.array(y_batch)}
				predict_results, loss, evals = sess.run([m.scores,m.total_loss, m.eval_op], feed_dict=feed)
				#print(predict_results)
				self.evals = evals

				
				sentences = []
				output_len = len(predict_results)
				with open(source_path,'r') as fin1:
					for line in fin1:
						sentences.append(line.strip())

				target = []
				with open(target_path,'r') as fin2:
					for line in fin2:
						v = line.strip().split()
						max_value = max(v)
						max_index = v.index(max_value)
						rel = self.relationtypes[max_index]
						target.append(rel)

				last_len = 0 - output_len

				sentences = sentences[last_len:]
				target = target[last_len:]

				
				out_path = test_path + '/system_results.txt'
				with open(out_path,'w') as fout:
					for i in range(len(predict_results)):
						res2 = [x for x in predict_results[0]]
						max_value = max(res2)
						max_index = res2.index(max_value)
						rel = self.relationtypes[max_index]
						s = sentences[i]
						e1_beg = s.find('<Q')
						e1_end = s.find('>',e1_beg)
						e1 = s[e1_beg:e1_end+1]
						e2_beg = s.find('<Q', e1_end)
						e2_end = s.find('>', e2_beg)
						e2 = s[e2_beg:e2_end+1]
						text = sentences[i] + '\t' + e1 + '\t' + e2 + '\t'+ rel + '\t' + target[i]
						fout.write(text+'\n')

				
				return out_path


	def get_model_data(self, source_path, target_path, *args, **kwargs):  
		train_data, test_data = util.read_data(source_path,target_path,self.sent_len,train_size = 100000,shuffle=False)
		return train_data,test_data

#def train(train_data, test_data, sent_len = None, vocab_size = None, num_classes = None):


def test_run():
	model = MLMI_CNN_Model()
	
	data_path = 'data/NYT/nyt_300.txt'
	model_data_path = model.read_dataset(data_path)
	model.data_preprocess(model_data_path)
	
	train_dir = model.train(model_data_path)
	print(train_dir)

	#train_dir = 'train/1556000498'
	evals = model.predict(model_data_path,trained_model = train_dir)
	model.evaluate(evals)



if __name__ == '__main__':
	test_run()
	#a = util.load_from_dump('train/1555207026/flags.cPickle')
	#print(a)

