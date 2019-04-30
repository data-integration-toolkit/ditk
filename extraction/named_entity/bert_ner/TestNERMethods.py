import unittest
import pandas as pd
from ner import Ner


class TestNERMethods(unittest.TestCase):

    def setUp(self):
        # self.ner = BertNer() #Your implementation of NER
        self.input_file = 'ner_test_input.txt'
        self.output_file = 'ner_test_output.txt'
        self.pred = Ner.predict(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter=' ')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.output_file)[0]
        input_col_count = self.row_col_count(self.output_file)[1]
        output_row_count = self.row_col_count(self.pred)[0]
        output_col_count = self.row_col_count(self.pred)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(input_col_count, output_col_count)

if __name__ == '__main__':
    unittest.main()