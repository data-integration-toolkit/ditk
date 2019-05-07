#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 02:12:14 2019

@author: aravindjyothi
"""

import unittest
from os import sys, path 
AtNRE_dir = path.dirname(path.dirname(path.abspath(__file__)))
src_dir = AtNRE_dir + '/src'
print(src_dir)
sys.path.append(src_dir)
sys.path.append(AtNRE_dir)

from main import AtNRE
obj = AtNRE()
    
#Please note that these tests will take time for the nyt corpus 
#Please also note that this invokes the training and testing process
#Make sure there are all the necessary data in /data, /cnndata, /origin_data, /result and so on before unit testing 
#This has not been uploaded to github due to space constraints, hence first train the model and then run tests

class TestImputationMethods(unittest.TestCase):

    def setUp(self):
        self.relation_extraction_method = obj  # The implementation of your Imputation method
        self.directory = AtNRE_dir + '/origin_data/train.txt'
        self.test_directory = AtNRE_dir + '/origin_data/test.txt'
        self.vec_dir = AtNRE_dir + '/origin_data/vec.txt'
        self.verificationErrors = [] #append exceptions for try-except errors
        

    def test_train_file_format(self):
        with open(self.directory, "r") as fin:
            lst = fin.readlines()
            total_lines = 1
            for line in lst:
                total_lines += 1
    
    def test_test_file_format(self):
        with open(self.test_directory, "r") as fin:
            lst = fin.readlines()
            total_lines = 1
            for line in lst:
                total_lines += 1
   
    def test_read_dataset(self):
        success_or_fail = self.relation_extraction_method.read_dataset(self.directory)
        success = 1
        self.assertEqual(success, success_or_fail)
        
    def test_load_embedding(self):
        success_or_fail = self.relation_extraction_method.load_embedding(self.vec_dir)
        success = 1
        self.assertEqual(success, success_or_fail)
            
    def test_data_preprocess(self):
        success_or_fail = self.relation_extraction_method.data_preprocess()
        success = 1
        self.assertEqual(success, success_or_fail)
        
    
    ## Run this test only after train, that is only after model is present! Uncomment this after training and test
    '''
    def test_predict(self):
        l_p, l_r= self.relation_extraction_method.predict()
        if len(l_p) and len(l_r):
            flag = 1
        success = 1
        self.assertEqual(success, flag)
    '''
    
    
if __name__ == '__main__':
	unittest.main()
