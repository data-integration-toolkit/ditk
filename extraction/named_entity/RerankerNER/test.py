import unittest
import reranker_predict
import pandas as pd

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        #self.ner = reranker_predict() #Your implementation of NER
        self.input_file = 'path_to_sample_input.txt'
        self.output_file = reranker_predict.main(input_file)

    def row_col_count(file_name):
        df = pd.read_csv(file_name,delim=' ')
        return df.shape

    def test_outputformat(self):    
        input_row_count = row_col_count(input_file)[0]
        input_col_count = row_col_count(input_file)[1]
        output_row_count = row_col_count(output_file)[0]
        output_col_count = row_col_count(output_file)[1]

        assertEqual(input_row_count, output_row_count)
        assertEqual(input_col_count + 1, output_col_count)

if __name__ == '__main__':
    unittest.main()