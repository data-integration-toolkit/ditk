import unittest
import pandas as pd
from model.main import DeepElmoEmbedNer


class TestNERMethods(unittest.TestCase):

    def setUp(self):
        self.ner = DeepElmoEmbedNer()
        self.input_file = '../data/sample/ner_test_input.txt'
        self.output_file = self.ner.main(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter=' ')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 3)


if __name__ == '__main__':
    unittest.main()
