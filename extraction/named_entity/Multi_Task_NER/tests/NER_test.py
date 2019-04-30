import unittest
import pandas as pd
import csv
from Multi_Task_NER import Multi_Task_NER


class TestNERMethods(unittest.TestCase):

    def setUp(self):

        self.ner = Multi_Task_NER() #Your implementation of NER
        self.train_file ='ner_test_input/train.txt'
        self.valid_file = 'ner_test_input/valid.txt'
        self.test_file = 'ner_test_input/test.txt'
        self.fileNames = {}
        self.fileNames['train'] = self.train_file
        self.fileNames['valid'] = self.valid_file
        self.fileNames['test'] = self.test_file
        self.output_file = self.ner.main(self.fileNames)


    def row_col_count(self,file_name):
        df = pd.read_csv(file_name, sep=' ', quoting=csv.QUOTE_NONE)
        return df.shape

    def test_outputformat(self):
        input_f = self.test_file
        output_f = self.output_file
        input_row_count = self.row_col_count(input_f)[0]
        input_col_count = self.row_col_count(input_f)[1]
        output_row_count = self.row_col_count(output_f)[0]
        output_col_count = self.row_col_count(output_f)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(3, output_col_count)

if __name__ == '__main__':
    unittest.main()