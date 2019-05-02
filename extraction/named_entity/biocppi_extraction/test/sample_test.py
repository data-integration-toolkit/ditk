import unittest
import pandas as pd
from biocppi_extraction import biocppi_extraction

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        self.ner = biocppi_extraction() #Your implementation of NER
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
