import unittest
import linecache
import pandas as pd
import os

from breds_relation_extraction import BREDSModel

class TestRelationExtraction(unittest.TestCase):

    def setUp(self):
        self.relation_extraction  = BREDSModel() 
        self.input_file = 'tests/sample_input.txt'
        self.model_data_path = self.relation_extraction.read_dataset(self.input_file)
        self.relation_extraction.data_preprocess(self.model_data_path)

        self.relation_extraction.train(self.input_file)

        self.output_file = self.relation_extraction.predict(self.model_data_path)
        self.truth_lable_file = 'tests/sample_output.txt'

        self.evals = self.relation_extraction.evaluate(self.output_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter='\t')
        return df.shape

    def test_read_dataset(self):
        self.assertTrue(os.path.exists(self.model_data_path))

    def test_predict(self):
        self.assertTrue(os.path.exists(self.output_file))

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.truth_lable_file)[0]
        input_col_count = self.row_col_count(self.truth_lable_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(input_col_count, output_col_count)

    def test_evaluate(self):
        self.assertEqual(len(self.evals),3)



if __name__ == '__main__':
    unittest.main()