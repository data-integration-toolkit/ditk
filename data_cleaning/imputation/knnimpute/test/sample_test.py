import csv
import unittest
import pandas as pd
import numpy as np
import os, sys
from os.path import isfile, join
from numpy import genfromtxt
sys.path.append("../..")
from imputation import Imputation
sys.path.append("..")
from main import knnImpute

#global object
knnimpute = knnImpute()

class TestImputationMethods(unittest.TestCase):
    def setUp(self):
        self.imputation_method = knnimpute
        self.input_file = "../../data/wdbc.csv"
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
            #input_headers = next(lines)
            total_input_lines = 0
            for line in lines:
                input_headers = len(line)
                total_input_lines += 1

        preprocess_result = self.imputation_method.preprocess(self.input_file)
        if isinstance(preprocess_result, list):
            for res in preprocess_result:
                if isinstance(res, DataLoader):
                    s = res.dataset.original_data.shape
                    self.assertEquals(s[0], total_input_lines)
                    self.assertEquals(s[1], input_headers)

    def test_evaluate(self, *args, **kwargs):
        # test if the evaluate function returns a numerical float value
        input = genfromtxt(self.input_file, delimiter=',')
        self.assertIsInstance(self.imputation_method.evaluate(trained_model = '', input = input, inputData = self.input_file, *args, **kwargs), float)

if __name__ == '__main__':
	unittest.main()
