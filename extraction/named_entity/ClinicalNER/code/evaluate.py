import os
import re

def evaluate_model(prediction_path, test_con_path):
	fp_predict = open(prediction_path, "r")
	fp_gold = open(test_con_path, "r")

	predict = fp_predict.read()
	gold = fp_gold.read()
	golds = gold.strip().split("\n")
	predicts = predict.strip().split("\n")
	dictionary = {}
	tp = 0
	fp = 0
	fn = len(golds)
	for concept in golds:
		ma = re.match(r"(c=\".*\") (\d*:\d* \d*:\d*)||t=(.*)", concept.strip())
		dictionary[ma.group(2)] = ma.group(3)

	for concept in predicts:
		ma = re.match(r"(c=\".*\") (\d*:\d* \d*:\d*)||t=(.*)", concept.strip())
		if (ma.group(2) in dictionary.keys()):
			fn -= 1
			if dictionary[ma.group(2)] == ma.group(3):
				tp += 1
			else:
				fp += 1
		else:
			fp += 1

	precision = tp * 1.0 / (tp + fp)
	recall = tp * 1.0 / (tp + fn)
	print ("Precision: ", precision)
	print ("Recall: ", recall)
	print ("F1 Score: ", 2 * precision * recall / (precision + recall))