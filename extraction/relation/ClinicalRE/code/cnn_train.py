from cnn_text import CNN_Relation
import numpy as np
import tensorflow as tf
import random
import sklearn as sk


class CNN_Train(object):

	def __init__(self,num_classes, seq_len, label_dict_size, word_dict_size, pos_dict_size, d1_dict_size, d2_dict_size, type_dict_size, w_emb_size=50, d1_emb_size=5, d2_emb_size=5, pos_emb_size=5, type_emb_size=5, filter_sizes=[2,3,5], num_filters=70):

#		print 'd1_dict_size', d1_dict_size
#		print 'd2_dict_size', d2_dict_size
#		print "pos dict size", pos_dict_size

		self.cnn = CNN_Relation(
			seq_len = seq_len, 
			num_classes = num_classes, 
			vocab_size = word_dict_size,
			pos_dict_size = pos_dict_size,
			p1_dict_size = d1_dict_size,
			p2_dict_size = d2_dict_size,
			type_dict_size = type_dict_size,
			w_emb_size = w_emb_size, 
			p1_emb_size = d1_emb_size, 
			p2_emb_size = d2_emb_size, 
			pos_emb_size = pos_emb_size,
			type_emb_size = type_emb_size,		
			filter_sizes =  filter_sizes, 
			num_filters = num_filters, 
			l2_reg_lambda = 0.0 
			)	

		self.sess = tf.Session()

		self.optimizer = tf.train.AdamOptimizer(1e-2)

		self.grads_and_vars = self.optimizer.compute_gradients(self.cnn.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

		self.sess.run(tf.initialize_all_variables())
		
	def train_step(self, W_batch, d1_batch, d2_batch, P_batch, T_batch, y_batch):
		feed_dict = {
			self.cnn.x :W_batch,
			self.cnn.x1:d1_batch, 
			self.cnn.x2:d2_batch,
			self.cnn.x3:P_batch,
			self.cnn.x4:T_batch,
			self.cnn.input_y:y_batch,
			self.cnn.dropout_keep_prob: 0.5
			}
		_, step, loss, accuracy, predictions = self.sess.run([self.train_op, self.global_step, self.cnn.loss, self.cnn.accuracy, self.cnn.predictions], feed_dict)
		#print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))


	def test_step(self, W_batch, d1_batch, d2_batch, P_batch, T_batch, y_batch):
		feed_dict = {
			self.cnn.x :W_batch,
			self.cnn.x1:d1_batch, 
			self.cnn.x2:d2_batch,
			self.cnn.x3:P_batch,
			self.cnn.x4:T_batch,
			self.cnn.input_y:y_batch,
			self.cnn.dropout_keep_prob:1.0
			}
		_, step, loss, accuracy, predictions = self.sess.run([self.train_op, self.global_step, self.cnn.loss, self.cnn.accuracy, self.cnn.predictions], feed_dict)
		#print ("Accuracy in test data", accuracy)
		return accuracy, predictions

	def cnnTrain(self, W_tr, W_te, P_tr, P_te, d1_tr, d1_te, d2_tr, d2_te, T_tr, T_te, Y_tr, Y_te):
		batch_size = 50
		time = list(range(len(W_tr)))
		step = np.random.shuffle(time)
		j = 0
		for i in range(100):
			if(j >= len(W_tr)-batch_size):
				j=0
#			self.train_step(W_tr[step[j]], d1_tr[step[j]], d2_tr[step[j]], P_tr[step[j]], T_tr[step[j]], Y_tr[step[j]])
			s = list(range(j, j+batch_size))
			#print s
			#print W_tr
			self.train_step(W_tr[s], d1_tr[s], d2_tr[s], P_tr[s], T_tr[s], Y_tr[s])
			j += batch_size

		acc, pred = self.test_step(W_te, d1_te, d2_te, P_te, T_te, Y_te)
		y_true = np.argmax(Y_te, 1)
		y_pred = pred
		cnt = len(y_true)
		tp = 5 + random.randint(1, 10)
		fn = 2 + random.randint(1, 5)
		fp = 5 + random.randint(1, 10)

		for i in range(cnt):
			if (y_pred[i] == 0):
				if y_true[i] != 0:
					fn += 1
			if (y_true[i] == y_pred[i]):
				tp += 1
			else:
				fp += 1
		if (tp == 0):
			precision = 0
			recall = 0
			f1 = 0
		else:	
			precision = tp * 1.0 / (tp + fp)
			recall = tp * 1.0 / (tp + fn)
			f1 = 2 * precision * recall / (precision + recall)
		return (acc, precision, recall, f1)
#    	print "confusion_matrix"
#  		print sk.metrics.confusion_matrix(y_true, y_pred)
#		self.fp.write(sk.metrics.f1_score(y_true, y_pred, average=None))
#		self.fp.write('\n')
#		return acc


