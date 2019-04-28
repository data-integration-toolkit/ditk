import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main


class TestEntityTypingMethods(unittest.TestCase):
    def setUp(self):
        self.input_file = 'ner_test_input.txt'
        self.output_file = main(self.input_file)

    def row_col_count(self, file_name, delimiter=' '):
        df = pd.read_csv(file_name, delimiter=delimiter)
        return df.shape

    def test_outputformat(self):
        output_col_count = self.row_col_count(self.output_file, '\t')[1]

        self.assertEqual(output_col_count, 2)


if __name__ == '__main__':
    unittest.main()
