from utils import *
from sklearn.model_selection import KFold
import numpy as np
from cnn_train import *

def train(fileName, label_dict, output_file):

	#ftrain = "data/coverted/DDI.txt"
	#ftrain = "data/coverted/SemEval.txt"
	#ftrain = "data/coverted/i2b2.txt"
	ftrain = fileName

	sent_contents, entity1_list, entity2_list, sent_lables = readData(ftrain)

	# Featurizer
	pos_tag_list = makePosFeatures(sent_contents)
	d1_list, d2_list, type_list = makeDistanceFeatures(sent_contents, entity1_list, entity2_list)

	#padding
	sent_contents,seq_len = makePaddedList(sent_contents)
	pos_tag_list,_ = makePaddedList(pos_tag_list)
	d1_list,_ = makePaddedList(d1_list)
	d2_list,_ = makePaddedList(d2_list)
	type_list,_ = makePaddedList(type_list)

	# Wordlist
	word_dict = makeWordList(sent_contents)
	pos_dict = makeWordList(pos_tag_list)
	d1_dict = makeWordList(d1_list)
	#print "d1_dict", d1_dict
	d2_dict = makeWordList(d2_list)
	type_dict = makeWordList(type_list)

	#vocabulary size
	word_dict_size = len(word_dict)
	pos_dict_size = len(pos_dict)
	d1_dict_size = len(d1_dict)
	d2_dict_size = len(d2_dict)
	type_dict_size = len(type_dict)
	label_dict_size = len(label_dict)

	#print "pos dict", pos_dict
	# Mapping
	W_train =  np.array(mapWordToId(sent_contents, word_dict))
	P_train = np.array(mapWordToId(pos_tag_list, pos_dict))
	d1_train = np.array(mapWordToId(d1_list, d1_dict))
	d2_train = np.array(mapWordToId(d2_list, d2_dict))
	T_train = np.array(mapWordToId(type_list,type_dict))

	Y_t = mapLabelToId(sent_lables, label_dict)
	Y_train = np.zeros((len(Y_t),label_dict_size))
	for i in range(len(Y_t)):
		Y_train[i][Y_t[i]] = 1.0

	"""
	print W_train.shape
	print P_train.shape 
	print d1_train.shape
	print d2_train.shape
	print T_train.shape
	print Y_train.shape

	"""

	kf = KFold(n_splits=10)
	acc_list = []
	precision_list = []
	recall_list = []
	f1_list = []
	cnt = 0
	for train, test in kf.split(W_train):
		W_tr, W_te = W_train[train], W_train[test]
		P_tr, P_te = P_train[train], P_train[test]
		d1_tr, d1_te = d1_train[train], d1_train[test]
		d2_tr, d2_te = d2_train[train], d2_train[test]
		T_tr, T_te = T_train[train], T_train[test]
		Y_tr, Y_te = Y_train[train], Y_train[test]

		train = CNN_Train(label_dict_size,
				seq_len, 		#length of largest sent
				label_dict_size, 	#number of classes
				word_dict_size,		#word vocab length
				pos_dict_size,		#pos vocab length
				d1_dict_size,		#d1 vocab length
				d2_dict_size,		#d2 vocab length
				type_dict_size)		#type vocab length	
				

		acc, precision, recall, f1 = train.cnnTrain(W_tr, W_te, P_tr, P_te, d1_tr, d1_te, d2_tr, d2_te, T_tr, T_te, Y_tr, Y_te)

		cnt += 1
		print ("Kold", cnt, ":")	
		print ("Precision: ", precision)
		print ("Recall: ", recall)
		print ("F1 Score: ", f1)
		print ("-------------------------------")

		precision_list.append(precision)
		recall_list.append(recall)

	p = np.mean(precision_list)
	r = np.mean(recall_list)
	print ("Precision: ", p)
	print ("Recall: ", r)
	print ("F1 Score: ", (p * r * 2.0) / (p + r))

	fp_output = open(output_file, "w")
	fp_output.write("Precision: " + str(p) + "\n")
	fp_output.write("Recall: " + str(r) + "\n")
	fp_output.write("F1 Score: " + str((p * r * 2.0) / (p + r)) + "\n")




