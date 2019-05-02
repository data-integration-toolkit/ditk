import unittest
import os
import sys
import pandas as pd
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestREMethods(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        import model
        self.input_file = os.path.abspath(os.getcwd() + '/testInput.txt')

        m = model.local_global_CRNN_bio(inputPath=self.input_file)

        # Read Raw Dataset
        m.read_dataset()

        # Split into Train: 90% Test: 10
        m.data_preprocess()

        # Parse with our group's Common Format and
        m.tokenize()

        # Train with our model
        m.train()

        # Generate Prediction File
        self.predict_file = m.predict()
        self.test_file = os.path.abspath(
            os.path.dirname(self.predict_file) + '/test.txt')

        # Evaluate with group's metric
        m.evaluate()

    def row_col_count(self, file_name):
        row_count, column_count = 0, 0
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip().rstrip('\n')
                it = line.split('\t')
                row_count += 1
                column_count = len(it)
        return row_count, column_count

    def test_outputformat(self):
        # We are goint to test output format with created test file
        print("test_file:", self.test_file)
        test_row_count, test_column_count = self.row_col_count(self.test_file)

        print("predict_file:", self.predict_file)
        predict_row_count, predict_column_count = self.row_col_count(
            self.predict_file)

        self.assertEqual(test_row_count, predict_row_count)
        self.assertEqual(test_column_count, 10)
        self.assertEqual(predict_column_count, 5)


if __name__ == '__main__':
    unittest.main()
