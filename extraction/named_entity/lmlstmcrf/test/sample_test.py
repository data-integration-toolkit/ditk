import sys, os

# set path to ditk root
ditk_path = os.path.abspath(os.getcwd())
main_path = os.path.abspath(os.path.join(sys.path[0], '..'))

if ditk_path not in sys.path:
    sys.path.append(ditk_path)
    sys.path.append(main_path)

data_dir = os.path.join(ditk_path, 'extraction/named_entity/lmlstmcrf/test/')
input_file = os.path.join(data_dir, 'sample_input.txt')
output_file = os.path.join(data_dir, 'sample_output.txt')

from extraction.named_entity.lmlstmcrf.hparams import hparams as hp 
hp.gpu = -1

import unittest
import pandas as pd
from extraction.named_entity.lmlstmcrf.lmlstmcrf import Lmlstmcrf 

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        self.ner = Lmlstmcrf() #Your implementation of NER
        self.input_file = input_file
        # self.output_file = self.ner.main(input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter=' ', error_bad_lines=False)
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(input_file)[0]
        input_col_count = self.row_col_count(input_file)[1]
        output_row_count = self.row_col_count(output_file)[0]
        output_col_count = self.row_col_count(output_file)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 3)

if __name__ == '__main__':
    unittest.main()