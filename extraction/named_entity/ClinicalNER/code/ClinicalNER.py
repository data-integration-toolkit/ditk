import sys
import os
import re
import evaluate
sys.path.append('code/')
import train
import predict
sys.path.remove('code/')
sys.path.append('../..')
from ner import Ner

class clinicalNER(Ner):

	def __init__(self):
		pass

	def convert_ground_truth(self, data, *args, **kwargs):
		pass

	def read_dataset(self, fileName, txt_path, con_path, *args, **kwargs):
		cnt = 0
		out_txt = open(txt_path, "w")
		out_con = open(con_path, "w")
		fp_input = open(fileName, "r")

		inputs = fp_input.read().strip().split('\n\n')

		for i in range(len(inputs)):

			input_words = inputs[i].strip().split('\n')
			for j in range(len(input_words)):
				input_word_attr = input_words[j].strip().split(' ');
				if j == 0:
					out_txt.write(input_word_attr[0])
				else:
					out_txt.write(" " + input_word_attr[0])
				if input_word_attr[3] == 'O':
					continue
				else:
					if input_word_attr[3][0] == 'B':
						start_pos = j
						entity = input_word_attr[0].lower()
					else:
						entity += " " + input_word_attr[0].lower()
					if ((j + 1) == len(input_words)) or input_words[j + 1].strip().split(' ')[3][0] != 'I':
						out_con.write("c=\"" + entity + "\" " + str(i + 1) + ":" + str(start_pos)
							+ " " + str(i + 1) + ":" + str(j) + "||t=\"" + input_word_attr[3][2:] + "\"\n")

			out_txt.write("\n")
		print ("Successfully read " + fileName + ".")

	def train(self, txt_path, con_path, model_path, *args, **kwargs):
		'''
		Args:
			data: Formatted data to be trained.
		Returns:
			None
		'''
		train.train_model(txt_path, con_path, model_path)
		print ("Training Completed.")

	def predict(self, model_path, test_txt_path, prediction_dir, *args, **kwargs):
		'''
		Args:
			data: Formatted test data to be predicted.
		Returns:
			predictions: [tuple,...], i.e. list of tuples.
		'''	
		predict.predict_model(model_path, test_txt_path, prediction_dir)
		fname = os.path.splitext(os.path.basename(test_txt_path))[0] + '.' + 'con'
		print ("Prediction Completed.")
		return prediction_dir + fname

	def output(self, test_txt_path, test_con_path, prediction_path, output_path, *args, **kwargs):
		fp_txt = open(test_txt_path, "r")
		fp_predict = open(prediction_path, "r")
		fp_gold = open(test_con_path, "r")
		fp_output = open(output_path, "w")

		txt = fp_txt.read()
		predict = fp_predict.read()
		gold = fp_gold.read()
		lines = txt.strip().split("\n")
		golds = gold.strip().split("\n")
		predicts = predict.strip().split("\n")
		outputs = []
		for line in lines:
			words = line.strip().split(" ")
			output = []
			for word in words:
				output.append([word, 'O', 'O'])
			outputs.append(output)

		for concept in golds:
			ma = re.match(r"c=\".*\" (\d*):(\d*) (\d*):(\d*)\|\|t=\"(.*)\"", concept.strip())
			line = int(ma.group(1))
			begin = int(ma.group(2))
			end = int(ma.group(4))
			con = ma.group(5)
			outputs[line - 1][begin][1] = 'B-' + con
			for i in range(begin + 1, end + 1):
				outputs[line - 1][i][1] = 'I-' + con

		for concept in predicts:			
			ma = re.match(r"c=\".*\" (\d*):(\d*) (\d*):(\d*)\|\|t=\"(.*)\"", concept.strip())
			line = int(ma.group(1))
			begin = int(ma.group(2))
			end = int(ma.group(4))
			con = ma.group(5)
			outputs[line - 1][begin][2] = 'B-' + con
			for i in range(begin + 1, end + 1):
				outputs[line - 1][i][2] = 'I-' + con

		for output in outputs:
			for word in output:
				fp_output.write(word[0] + ' ' + word[1] + ' ' + word[2] + '\n')
			fp_output.write('\n')
		print ("Write to output files.")

	def evaluate(self, prediction_path, test_con_path, *args, **kwargs):
		'''
		Args:
            predictions: [tuple,...], list of tuples [same format as output from predict]
            groundTruths: [tuple,...], list of tuples representing ground truth.

        Returns:
            metrics: tuple with (p,r,f1). Each element is float.
		'''
		evaluate.evaluate_model(prediction_path, test_con_path)
		print ("Evaluation Completed.")

	def save_model(self, file):
		'''
		:param file: Where to save the model - Optional function
		:return:
		'''
		pass

	def load_model(self, file):
		'''
		:param file: From where to load the model - Optional function
		:return:
		'''
		pass
