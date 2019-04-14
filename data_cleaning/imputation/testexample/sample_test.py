import csv
import unittest
import pandas as pd
import numpy as np

from data_cleaning.imputation import Imputation

class TestImputationMethods(unittest.TestCase):

	def setUp(self):
		self.imputation_method = "impute_method"  # The implementation of your Imputation method
		self.input_file = "path to the input filename"
		self.verificationErrors = []

	def test_input_file_format(self):
		# test if input file agrees with expected format
		try:
			with open(self.input_file, "r") as fin:
				lines = csv.reader(fin)
				total_lines = 0
				for line in lines:
					total_lines += 1
		except Exception as e:
			self.verificationErrors.append(e)

	def test_preprocess(self):
		preprocess_result = self.imputation_method.preprocess(self.input_file)

		try: self.assertIsInstance(preprocess_result, pd.DataFrame) # Check if type is Data Frame
		except AssertionError as e: self.verificationErrors.append(e)
		try: self.assertIsInstance(preprocess_result, np.ndarray) # Check if type is Numpy
		except AssertionError as e: self.verificationErrors.append(e)
		try:
			with open(preprocess_result, "r") as fin:
				lines = csv.reader(fin)
				total_lines = 0
				for line in lines:
					total_lines = 0
				self.verificationErrors.append("Non Empty ")



	def test_predict_format(self):
		prediction_score = tss.predict(data_X, data_Y)
		assertTrue(isFloat(prediction_score))
		assertTrue(prediction_score >= 0 and prediction_score <= 1.0)
		with self.assertRaises(ValueError):
			tss.predict(empty_list)  # Fails on wrong input format

	def test_predict_similar(self):
		prediction_score = tss.predict(data_X, data_X)
		assertTrue(prediction_score == 1.0)  # Same sentences

	def test_predict_dissimilar(self):
		prediction_score = tss.predict(data_X, data_Y)
		assertTrue(prediction_score < 1.0)  # Dissimilar sentences


if __name__ == '__main__':
	unittest.main()
