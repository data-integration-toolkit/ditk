import unittest
import pandas as pd
import relation


class TestRelationExtraction(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        self.relation_extraction = RelationExtraction()
        self.input_file = 'relation_extraction_test_input.txt'
        self.output_file = 'relation_extraction_test_output.txt'
        self.predic = relation_extraction.predict(self.input_file)

    def row_col_count(self, file_name):
        df = pd.read_csv(file_name, delimiter='\t')
        return df.shape

    def test_outputformat(self):
        input_row_count = self.row_col_count(self.output_file)[0]
        input_col_count = self.row_col_count(self.output_file)[1]
        output_row_count = self.row_col_count(self.pred)[0]
        output_col_count = self.row_col_count(self.pred)[1]

        self.assertEqual(input_row_count, output_row_count)
        self.assertEqual(input_col_count, output_col_count)


if __name__ == '__main__':
    unittest.main()