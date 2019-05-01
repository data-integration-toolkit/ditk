import csv
import unittest
import pandas as pd
import numpy as np
import sys, os

#give path to refactored code
sys.path.insert(0,"/Users/tushyagautam/Documents/USC/Information_Integration/Project/HMF_Submission/")


from nmf_np_ref import HMF_Class

class TestImputationMethods(unittest.TestCase):

	def setUp(self):
		self.imputation_method = HMF_Class()  # The implementation of your Imputation method
		self.input_file = "input file name here"
		self.verificationErrors = [] # append exceptions for try-except errors
		#provide the path to the sample input file
		self.inputData = "path to sample input file here"

	def test_input_file_format(self):
		# test if input file agrees with expected format
		with open(self.input_file, "r") as fin:
			lines = csv.reader(fin)
			total_lines = 0
			for line in lines:
				total_lines += 1

	def test_impute(self):

		# Test whether the final imputed data have the same shape with input data
		with open(self.input_file, "r") as fin:
			lines = csv.reader(fin)
			input_headers = next(lines)
			total_input_lines = 0
			for line in lines:
				total_input_lines += 1

		self.imputation_method.preprocess(self.inputData, self.input_file)
		preprocess_result = self.imputation_method.impute()
		if isinstance(preprocess_result, np.ndarray):
			self.assertEquals(preprocess_result.shape[0], total_input_lines+1)
			# self.assertEquals(preprocess_result.shape[1], input_headers)
		elif isinstance(preprocess_result, pd.DataFrame):
			self.assertEquals(preprocess_result.shape[0], total_input_lines)
			self.assertEquals(preprocess_result.shape[1], input_headers)
		else:
			with open(self.input_file, "r") as fin:
				lines = csv.reader(fin)
				output_headers = next(lines)
				total_output_lines = 0
				for line in lines:
					total_output_lines += 1
			self.assertEquals(total_input_lines, total_output_lines)
			# self.assertEquals(input_headers, output_headers)



if __name__ == '__main__':
	unittest.main()
