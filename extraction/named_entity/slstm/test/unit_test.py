import unittest
import pandas as pd
import sys
sys.path.append("..")
from model.config import Config
import final_refact_slstm
import tensorflow as tf
import csv


from final_refact_slstm import ner_extraction


class TestNERMethods(unittest.TestCase):



    def setUp(self):
        config = Config()
        config.layer=int(20) #iterations
        config.step=int(1)

        self.ner = ner_extraction(config)
        self.input_file = '../ner_test_input.txt'
        self.output_file = final_refact_slstm.main(self.input_file)
        # self.output_file = main(self.input_file)

    def row_count(self, file_name): 
        return sum(1 for line in open(file_name))
 

    def col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter=' ',error_bad_lines=False)
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_count(self.input_file)
        output_row_count = self.row_count(self.output_file)
        output_col_count = self.col_count(self.output_file)[1]-1

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 3)


if __name__ == '__main__':
    unittest.main()