import unittest
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import csv

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        
        self.input_file = './test/conll2003_sample.txt'# conll2003 data sample
        # self.input_file = "./test/ontonotes_sample.txt" 
        self.output_file = main(self.input_file)
        # self.output_file = "./data/conll.test.output" # REMOVE 

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter=' ', error_bad_lines=False, header=None)
        return df.shape

    def test_outputformat(self):  
        print("input_file:", self.input_file)
        print("output_file:", self.output_file)
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        print(input_row_count, ",", input_col_count)
        print(output_row_count, ",", output_col_count)

        self.assertEqual(input_row_count, output_row_count-1)
        self.assertEqual(output_col_count, 2) 


if __name__ == '__main__':
    unittest.main()
