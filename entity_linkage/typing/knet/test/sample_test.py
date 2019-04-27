import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


class TestEntityTypingMethods(unittest.TestCase):
    def setUp(self):
        self.input_file = 'entity_typing_test_input.txt'
        self.output_file = main(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter='\t')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(input_col_count, output_col_count + 1)


if __name__ == '__main__':
    unittest.main()
