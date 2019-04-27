import unittest
import pandas as pd
from typing import entity_typing 

def row_col_count(file_name):
    df = pd.read_csv(file_name, delimiter='\t')
    return df.shape

class TestEntityTypingMethods(unittest.TestCase):

    def setUp(self):
        self.et = entity_typing() #Your implementation of Entity Typing
        self.input_file = 'entity_typing_test_input.txt'
        self.output_file = et.main(input_file)

    def test_outputformat(self):    
        input_row_count = row_col_count(self.input_file)[0]
        input_col_count = row_col_count(self.input_file)[1]
        output_row_count = row_col_count(self.output_file)[0]
        output_col_count = row_col_count(self.output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(input_col_count, output_col_count + 1)


if __name__ == '__main__':
    unittest.main()
