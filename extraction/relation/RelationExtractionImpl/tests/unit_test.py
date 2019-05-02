import numpy
import unittest
import pandas as pd
from relation.RelationExtractionImpl.src.RelationExtraction_impl import DDIExtractionImpl #my implementation


class TestNERMethods(unittest.TestCase):
    def setUp(self):
        self.RelationExtraction = DDIExtractionImpl() #Your implementation of NER
        self.input_file = "relation_extraction_test_input.txt"
        self.output_file =self.RelationExtraction.main(self.input_file)
    def row_col_count(self,file_name):
        df = pd.read_csv(file_name,delimiter='\t')
        return df.shape
    def test_outputformat(self):
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]
        # print(input_row_count)
        # print(output_row_count)
        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 5)
if __name__ == '__main__':
    unittest.main()