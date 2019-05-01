import os
from svm_semantic_similarity import svm_semantic_similarity as sts
from gensim.models import KeyedVectors

def main()


if __name__ == '__main__':
		# args = parser.parse_args()
		# pp.pprint(args)

	print("Loading word2vec...")
	# Load the training data
	vecfile = 'GoogleNews-vectors-negative300.bin'
	vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)

	# with open(args.inputfile, 'r') as inputfile:
	#     input = inputfile.readlines()
		
	print("reading input....")


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


# 	train_input = "./data/sts2014/train.txt"

# 	val1 = "./data/sts2014/val.txt"

# 	test1 = "./data/sts2014/test.txt"


	

# 	# with open(val1, 'r') as val_file:
# 	# 	val = val_file.readlines()


# 	# with open(test1, 'r') as test_file:
# 	# 	test = test_file.readlines()

# 	sss = SVM_SemsSim(vecs)
# # 	test_input = open('similarity_test_input.txt','r').readlines()
# # 	for pair in test_input:
# # 		s1,s2 = pair.split(",")
# # 		pred = sss.predict(s1,s2)
# # 		print(pred[0]/5.0)

# 	input_list = sss.read_dataset(train_input)
# 	train_feats = sss.generate_embeddings(input_list, vecs)


# 	train_sims = []

# 	for rec in input_list:
# 		train_sims.append(float(rec[2]))

	sss.train(train_feats,train_sims)
	
	sss.save_model('test_model.pkl')
	
	model = sss.load_model('test_model.pkl')

	test_list = sss.read_dataset(test1)
	test_sims = []

	for rec in test_list:
		test_sims.append(float(rec[2]))
	test_feats = sss.generate_embeddings(test_list,vecs)

	pred = sss.predict(test_feats,model)

	eval = sss.evaluate(test_sims,pred)

	print("Pearson Correlation Coefficient: ", eval)

    
		
		

		# Baseline
		# generates prediction for input textfile, 'pred_simple.txt'
		# simple_baseline(input, output_simple)

		# # Extension 1
		# # generates prediction for input textfile, 'pred_ex1.txt'
		# w2v_semantic_sim(input, vecs, output_w2v)


		# print("training model....")
		# # Extension 2 + More [tried several different models/features to achieve best results --> see writeup]
		# # automatically generates predictions for validation and test set, 'pred_val_ex2.txt' and 'pred_test_ex2.txt'
		# supervised_synonym_sim(train_input, val, test, vecs)
	
  
	'''
	Sample Workflow:
	
	inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

	embeddings = genereate_embeddings(input text)
	
	model = train(train_input, val, test, vecs, output_file)
	
	similarity = predict(model)
	
	pearson_correlation = evaluate(test_labels, similarity)
	
	'''