import unittest
import pandas as pd
import os, sys
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        
        self.input_file = './test/conll2003_sample.txt' # conll2003 data sample
        # self.input_file = "./test/ontonotes_sample.txt" # ontonotes data sample
        self.output_file = main(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter=' ', quotechar=None, quoting=3)
        return df.shape

    def row_count(self, file_name):
        df = pd.read_csv(file_name, delimiter='\n', quotechar=None, quoting=3)
        return df.shape[0]

    def test_outputformat(self):  

        if "conll2003" in self.input_file:
            print("input_file:", self.input_file)
            input_row_count = self.row_col_count(self.input_file)[0]
            input_col_count = self.row_col_count(self.input_file)[1]
            # print(input_row_count, ",", input_col_count)

            print("output_file:", self.output_file)
            output_row_count = self.row_col_count(self.output_file)[0]
            output_col_count = self.row_col_count(self.output_file)[1]
            # print(output_row_count, ",", output_col_count)

            self.assertEqual(input_row_count, output_row_count)
            self.assertEqual(input_col_count-1, output_col_count)

        elif "ontonotes" in self.input_file:
            print("input_file:", self.input_file)
            input_row_count = self.row_count(self.input_file)
            # print(input_row_count, ", .")

            print("output_file:", self.output_file)
            output_row_count = self.row_col_count(self.output_file)[0]
            output_col_count = self.row_col_count(self.output_file)[1]
            # print(output_row_count, ",", output_col_count)

            self.assertEqual(input_row_count, output_row_count)
            self.assertEqual(output_col_count, 3)


if __name__ == '__main__':
    unittest.main()
