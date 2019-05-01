import unittest
import pandas as pd
from disease_name_recognition_through_rnn import disease_name_recognition_through_rnn

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        self.ner = disease_name_recognition_through_rnn() #Your implementation of NER
        self.input_file = 'test/sample_input.txt'
        self.output_file = self.ner.unittest_main(self.input_file)

    def row_col_count(self,file_name):
        df = pd.read_csv(file_name,delimiter=' ')
        return df.shape

    def test_outputformat(self):
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        self.assertEqual(output_col_count, 3)

if __name__ == '__main__':
    unittest.main()
