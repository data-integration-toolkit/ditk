import csv
import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append("../src")
from mice import Mice
sys.path.append("../..")
from imputation import Imputation

#global object
imputer = Mice(n_iter=5, sample_posterior=True, random_state=3)

class TestImputationMethods(unittest.TestCase):

	def setUp(self):
		self.imputation_method = imputer  # The implementation of your Imputation method
		self.input_file = "imputation_test_input.csv"
		self.output_file = "imputation_test_output.csv"
		self.verificationErrors = [] # append exceptions for try-except errors

	def test_input_file_format(self):
		# test if input file agrees with expected format
		with open(self.input_file, "r") as fin:
			lines = csv.reader(fin)
			total_lines = 1
			for line in lines:
				total_lines += 1

	def test_preprocess_file_format_and_missingness(self):
		# Test whether the preprocess data have the same shape with input data and that it introduces missingness.
		with open(self.input_file, "r") as fin:
			lines = csv.reader(fin)
			input_headers = len(next(lines))
			total_input_lines = 1
			for line in lines:
				total_input_lines += 1
		"""
		check format with input
		"""
		preprocess_result = self.imputation_method.preprocess(self.input_file)
		row = preprocess_result.shape[0]
		column = preprocess_result.shape[1]
		if isinstance(preprocess_result, np.ndarray):
			self.assertEqual(row, total_input_lines)
			self.assertEqual(column, input_headers)
		elif isinstance(preprocess_result, pd.DataFrame):
			self.assertEqual(row, total_input_lines)#since in dataframe, row and column start with 0.
			self.assertEqual(column, input_headers)#since in dataframe, row and column start with 0.
		else:
			with open(self.input_file, "r") as fin:
				lines = csv.reader(fin)
				output_headers = len (next(lines))
				total_output_lines = 1
				for line in lines:
					total_output_lines += 1
			self.assertEqual(total_input_lines, total_output_lines)
			self.assertEqual(input_headers, output_headers)

# Test whether the preprocess data has at least one value as null to check that it introduces missingness.
		null_preprocess=preprocess_result.isnull()
		a=0; b=0
		for i in range(row):
			for j in range(column):
					if null_preprocess.at[i,j] == True: 
						a=i;b=j
						break
		self.assertEqual(null_preprocess.at[a,b],True)

	def test_impute(self):
# Test whether the imputed result data has none of the values as null.
		preprocess_result = self.imputation_method.preprocess(self.input_file)
		row = preprocess_result.shape[0]
		column = preprocess_result.shape[1]
		imputed_table = self.imputation_method.impute(preprocess_result)
		imputed_table_dataframe = round (pd.DataFrame(imputed_table),2)#rounding values to 2 decimal places
		imputed_table_dataframe.to_csv(self.output_file,header=None,index=None)#output dataframe to csv file
		"""
		check if imputed table has same row and column length
		"""
		row_impute = imputed_table.shape[0]
		column_impute = imputed_table.shape[1]
		self.assertEqual(row,row_impute)
		self.assertEqual(column,column_impute)
		"""
		check if the impute gives complete data frame

		"""
		for i in range (row):
			for j in range (column):
					self.assertNotEqual(imputed_table[i][j],None) #none of the values should be null.


if __name__ == '__main__':
	unittest.main()