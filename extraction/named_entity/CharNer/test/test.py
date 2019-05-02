
import unittest
import pandas as pd
from src.main import CharNER
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class TestNERMethods(unittest.TestCase):

    def setUp(self):
        self.ner = CharNER() #Your implementation of NER
        self.input_file = "../data/sample_common_input.txt"
        self.output_file =self.ner.main(self.input_file)


    def row_col_count(self,file_name):
        df = pd.read_csv(file_name,delimiter=' ')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.input_file)[0]
        input_col_count = self.row_col_count(self.input_file)[1]
        output_row_count = self.row_col_count(self.output_file)[0]
        output_col_count = self.row_col_count(self.output_file)[1]

        print(input_row_count)
        print(output_row_count)

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(output_col_count, 3)

if __name__ == '__main__':
    unittest.main()