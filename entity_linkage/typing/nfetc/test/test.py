import unittest
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
import csv

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        
        self.input_file = "./test/clean_data.tsv"
        # self.input_file = "./test/raw_data.txt"
        self.output_file = main(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter='\t')
        return df.shape

    def test_outputformat(self):  
        print("input_file:", self.input_file)
        print("output_file:", self.output_file)
        input_num = self.row_col_count(self.input_file)
        output_num = self.row_col_count(self.output_file)
        input_row_count = input_num[0]
        input_col_count = input_num[1]
        output_row_count = output_num[0]
        output_col_count = output_num[1]

        print(input_row_count, ",", input_col_count)
        print(output_row_count, ",", output_col_count)

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 2)
        self.assertEqual(input_col_count, 5)

if __name__ == '__main__':
    unittest.main()
