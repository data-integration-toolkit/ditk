import unittest
import pandas as pd

from src.main import main


class TestRelationExtraction(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        # self.relation_extraction = main()
        self.input_file = '../data/relation_extraction_test_input.txt'
        self.output_file = main(self.input_file)


    def row_col_count(self,file_name):
        df = pd.read_csv(file_name,delimiter='\t')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]


        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 5)


if __name__ == '__main__':
    unittest.main()