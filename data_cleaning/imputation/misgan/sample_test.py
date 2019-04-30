import csv
import unittest
import os
from os.path import isfile, join
import sys
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from main import MisGAN

class Args():
    def __init__(self):
        self.input = "..\\data\\wdbc.csv"
        self.fname = "data/wdbc.csv"
        self.ims = True
        self.preprocess = True
        self.evaluate = True
        self.split = False
        self.model = "wdbc.csv_train"


class TestImputationMethods(unittest.TestCase):
    def setUp(self):
        self.args = Args()
        self.imputation_method = MisGAN(self.args)  # The implementation of your Imputation method
        self.input_file = self.imputation_method.args.input
        self.verificationErrors = []  # append exceptions for try-except errors

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
            total_input_lines = 0
            for l in lines:
                input_headers = len(l)
                total_input_lines += 1

        preprocess_result = self.imputation_method.preprocess()

        if isinstance(preprocess_result, list):
            for res in preprocess_result:
                if isinstance(res, DataLoader):
                    s = res.dataset.original_data.shape
                    self.assertEquals(s[0], total_input_lines)
                    self.assertEquals(s[1], input_headers)


    def test_evaluate(self, *args, **kwargs):
        self.assertIsInstance(self.imputation_method.evaluate(*args, **kwargs), float)


if __name__ == '__main__':
    unittest.main()
