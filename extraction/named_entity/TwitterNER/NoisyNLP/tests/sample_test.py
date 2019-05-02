import unittest
import pandas as pd
from twitterner import TwitterNER 

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        self.ner = TwitterNER() #Your implementation of NER
        self.input_file = 'Ner_test_input.txt'
        self.output_file = ner.main(input_file)

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
