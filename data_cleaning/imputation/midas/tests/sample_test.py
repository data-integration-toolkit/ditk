import csv
import unittest
import pandas as pd
import numpy as np

from data_cleaning.imputation import Imputation

class TestImputationMethods(unittest.TestCase):

	def setUp(self):
		self.imputation_method = "impute_method"  # The implementation of your Imputation method
		self.input_file = "path to the input filename"
		self.verificationErrors = [] # append exceptions for try-except errors

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

		preprocess_result = self.imputation_method.preprocess(self.input_file)
		if isinstance(preprocess_result, np.ndarray):
			self.assertEquals(preprocess_result.shape[0], total_input_lines)
			self.assertEquals(preprocess_result.shape[1], input_headers)
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
			self.assertEquals(input_headers, output_headers)


	def test_evaluate(self, *args, **kwargs):
		self.assertIsInstance(self.imputation_method.evaluate(*args, **kwargs), float)


if __name__ == '__main__':
	unittest.main()