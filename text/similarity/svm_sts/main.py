# -*- coding: utf-8 -*-
import os,sys
from svm_semantic_similarity import svm_semantic_similarity
from gensim.models import KeyedVectors

def main(input_path):
	print("Loading word2vec...")

	#path to google vectors file
	#download and place it in the same folder
	vecfile = 'GoogleNews-vectors-negative300.bin'
	vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)

	path = os.getcwd()

	train_input = path+ "/data/sts2014/train.txt"

	val1 = path + "/data/sts2014/val.txt"

	test1 = path+ "/data/sts2014/test.txt"

		
	print("reading input....")
	file = open(input_path,"r").readlines()
	sent1 = file[0].strip().replace("\"","")
	sent2 = file[1].strip().replace("\"","")
	print("Sentence1:" + sent1)
	print("Sentence2:" + sent2)
	
	sts = svm_semantic_similarity(vecs)

	input_list = sts.read_dataset(train_input)
	train_feats = sts.generate_embeddings(input_list, vecs)

	train_sims = []

	for rec in input_list:
		train_sims.append(float(rec[2]))

	sts.train(train_feats,train_sims)

	sts.save_model('test_model.pkl')

	model = sts.load_model('test_model.pkl')

	res = round(sts.predict_score(sent1,sent2)[0],3)/5.0

	print(res)

	# output_simple = open("pred_simple.txt", "w")
	# output_w2v = open("pred_ex1.txt", "w")


	#SemEval 2017
	# train_input = "/Users/aishwaryasp/Desktop/semantic_textual_similarity-master/data/en-train.txt"

	# val = "/Users/aishwaryasp/Desktop/semantic_textual_similarity-master/data/en-val.txt"

	# test = "/Users/aishwaryasp/Desktop/semantic_textual_similarity-master/data/en-test.txt"

	# #SICK 2014

	# train_input = "/Users/aishwaryasp/Desktop/SICK2014/SICK_train.txt"

	# val1 = "/Users/aishwaryasp/Desktop/SICK2014/SICK_trial.txt"

	# test1 = "/Users/aishwaryasp/Desktop/SICK2014/SICK_test_annotated.txt"


	


	

	# with open(val1, 'r') as val_file:
	# 	val = val_file.readlines()


	# with open(test1, 'r') as test_file:
	# 	test = test_file.readlines()

	# sss = SVM_SemsSim(vecs)
	# 	test_input = open('similarity_test_input.txt','r').readlines()
	# 	for pair in test_input:
	# 		s1,s2 = pair.split(",")
	# 		pred = sss.predict(s1,s2)
	# 		print(pred[0]/5.0)

	# input_list = sss.read_dataset(train_input)
	# train_feats = sss.generate_embeddings(input_list, vecs)


	# train_sims = []

	# for rec in input_list:
	# 	train_sims.append(float(rec[2]))

	# sss.train(train_feats,train_sims)
	
	# sss.save_model('test_model.pkl')
	
	# model = sss.load_model('test_model.pkl')

	# test_list = sss.read_dataset(test1)
	# test_sims = []

	# for rec in test_list:
	# 	test_sims.append(float(rec[2]))
	# test_feats = sss.generate_embeddings(test_list,vecs)

	# pred = sss.predict(test_feats,model)

	# eval = sss.evaluate(test_sims,pred)

	# print("Pearson Correlation Coefficient: ", eval)

	

if __name__ == '__main__':
		# args = parser.parse_args()
		# pp.pprint(args)
	
	main("./tests/sample_input.txt")
		

	'''
	Sample Workflow:
	
	inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

	embeddings = genereate_embeddings(input text)
	
	model = train(train_input, val, test, vecs, output_file)
	
	similarity = predict(model)
	
	pearson_correlation = evaluate(test_labels, similarity)
	
	'''