import unittest
import pandas as pd
import main
import os
import sys

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        self.input_file = os.path.join(os.getcwd(), 'ner_test_input.txt')
        self.output_file = main.main(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name,delimiter=' ')
        return df.shape

    def test_outputformat(self):
        output_col_count = self.row_col_count(self.output_file)[1]
        self.assertEqual(output_col_count, 3)

if __name__ == '__main__':
    unittest.main()
