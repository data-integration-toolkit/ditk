import unittest
import pandas as pd
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import main
# from transferlearning import main

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
