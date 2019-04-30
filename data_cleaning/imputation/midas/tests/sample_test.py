import csv
import unittest
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
sys.path.append("../src")
from midas import Midas
sys.path.append("../..")
from imputation import Imputation

#global object
imputer = Midas(layer_structure= [128], vae_layer= False, seed= 42)

class TestImputationMethods(unittest.TestCase):

	def setUp(self):
		self.imputation_method = imputer  # The implementation of your Imputation method
		self.input_file = "imputation_test_input.csv"
		self.output_file = "imputation_test_output.csv"
		self.no_of_imputations = 5 
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
		"""
		check preprocess data's format is same as input
		"""
		with open(self.input_file, "r") as fin:
			lines = csv.reader(fin)
			input_headers = len(next(lines))
			total_input_lines = 1
			for line in lines:
				total_input_lines += 1

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
# Test imputed result data with three checks - file format, all values not none, no of imputations
		preprocess_result = self.imputation_method.preprocess(self.input_file)
		row = preprocess_result.shape[0]
		column = preprocess_result.shape[1]

		scaler = MinMaxScaler() #scale to [0,1]
		preprocess_result = pd.DataFrame(scaler.fit_transform(preprocess_result), columns= preprocess_result.columns)
		columns_list = []
		self.imputation_method.build_model(preprocess_result, softmax_columns= columns_list)
		self.imputation_method.overimpute(training_epochs= 100, report_ival= 1,report_samples= 5, plot_all= False)
		self.imputation_method.train(train_data = imputer.imputation_target.values,training_epochs= 100, verbosity_ival= 1)
		self.imputation_method.impute(self.no_of_imputations)
		"""
		Check if no. of imputated values in output list is same as the given no. of imputations
		"""
		self.assertEqual(len(imputer.output_list),self.no_of_imputations)
		"""
		check if imputed table has same row and column length as the preprocess data
		"""
		imputed_table=imputer.output_list[0]
		imputed_table=pd.DataFrame(scaler.inverse_transform(imputed_table))  ##scaler inverse
		imputed_table = round(imputed_table,2)
		imputed_table.to_csv(self.output_file,header=None,index=None)
		row_impute = imputed_table.shape[0]
		column_impute = imputed_table.shape[1]
		self.assertEqual(row,row_impute)
		self.assertEqual(column,column_impute)
		"""
		check if the impute gives complete data frame

		"""
		for i in range (row_impute):
			for j in range (column_impute):
					self.assertNotEqual(imputed_table.at[i,j],None) #none of the values should be null.


if __name__ == '__main__':
	unittest.main()