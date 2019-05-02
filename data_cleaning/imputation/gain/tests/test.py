#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:20:01 2019

@author: aravindjyothi
"""

import csv
import unittest

from os import sys, path 


imputation_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
module_dir = path.dirname(path.dirname(path.abspath(__file__)))

src_dir = module_dir + '/src'
data_dir = module_dir +'/data'

sys.path.append(src_dir)
sys.path.append(module_dir)


#add /gain/src visibility to system in order to import 
#test.py is in tests 


from GAIN import GAIN
from os import path 

#method object
gain_obj = GAIN(128, 0.2, 0.9, 10, 0.8) 

class TestImputationMethods(unittest.TestCase):
    def setUp(self):
        self.imputation_method = gain_obj  # The implementation of your Imputation method
        self.input_file = data_dir + "/Letter.csv" #Change dataset here
        self.output_file = module_dir + "/output.csv" # Output is in /gain directory
        self.verificationErrors = [] #append exceptions for try-except errors
        self.row_size = None
        self.col_size = None
    
    def test_input_file_format(self):
        with open(self.input_file, "r") as fin:
            lines = csv.reader(fin)
            total_lines = 1
            for line in lines:
                total_lines += 1
   
    def test_preprocess(self):
        with open(self.input_file, "r") as fin:
            lines = csv.reader(fin)
            input_headers = len(next(lines))
            total_lines = 1
            for line in lines:
                total_lines += 1
        preprocess_result, no , dim = self.imputation_method.preprocess(self.input_file)
        self.row_size = no 
        self.col_size = dim
        self.assertEqual(no, total_lines)
        self.assertEqual(dim, input_headers)
    
    def test_introduce_missingness(self):
        with open(self.input_file, "r") as fin:
            lines = csv.reader(fin)
            input_headers = len(next(lines))
            total_lines = 1
            for line in lines:
                total_lines += 1
        preprocess_result, no , dim = self.imputation_method.preprocess(self.input_file)
        missingness_matrix = self.imputation_method.introduce_missingness(dim,no, preprocess_result)
        m_row, m_col = missingness_matrix.shape[0],  missingness_matrix.shape[1]
        self.assertEqual(m_row, total_lines)
        self.assertEqual(m_col, input_headers)
    
    def test_impute(self):
        res, no, dim= self.imputation_method.preprocess(self.input_file)
        self.imputation_method.impute()
        with open(self.output_file, "r") as fout:
            lines = csv.reader(fout)
            input_headers = len(next(lines))
            total_lines = 1
            for line in lines:
                total_lines += 1
        self.assertEqual(no, total_lines)
        self.assertEqual(dim, input_headers)

if __name__ == '__main__':
	unittest.main()