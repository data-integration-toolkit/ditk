import os
import sys
import unittest

import pandas as pd

PACKAGE_PARENT = '../../../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
print(os.getcwd())
from extraction.named_entity.cvt.cross_view import CrossViewTraining
from extraction.named_entity.cvt import main


class TestNERMethods(unittest.TestCase):

    def test_a_setup(self):
        self.file_root = os.getcwd()
        print("files root is: ", self.file_root)
        self.input_file = self.file_root + '/tests/sample_input.txt'
        self.output_file = ''
        self.ner = CrossViewTraining()
        self.output_file = main.main(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delim=' ')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 3)
        self.assertEqual(input_col_count, 17)


if __name__ == '__main__':
    unittest.main()
