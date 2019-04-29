import unittest
import linecache
import pandas as pd
from configure import FLAGS

from lstm_relation_extraction import LSTM_relation_extraction

class TestRelationExtraction(unittest.TestCase):

    def setUp(self):
        FLAGS.embeddings = ''
        FLAGS.data_type = 'semeval2010'
        self.relation_extraction  = LSTM_relation_extraction()  # Your implementation of NER
        self.input_file = 'testexample/sample_input.txt'
        self.output_file = self.relation_extraction.main(self.input_file)
        self.truth_lable_file = 'testexample/sample_output.txt'

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter='\t')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.truth_lable_file)[0]
        input_col_count = self.row_col_count(self.truth_lable_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(input_col_count, output_col_count)



if __name__ == '__main__':
    unittest.main()