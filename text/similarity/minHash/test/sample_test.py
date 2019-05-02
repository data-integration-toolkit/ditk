import unittest
import pandas as pd
from MinHash import MinHash

class TestTextSimilarityMethods(unittest.TestCase):

    def setUp(self):
        self.tss = MinHash()
        self.data_X = "This is a sample sentence to test text similarity"
        self.data_Y = "This is a different sentence to test text similarity"
        self.empty_list = []

    def test_predict_format(self):
        prediction_scores = self.tss.predict(self.data_X, self.data_Y)
        self.assertTrue(prediction_scores >= 0 and prediction_scores <= 1.0)
        with self.assertRaises(TypeError):
            self.tss.predict(self.empty_list) #Fails on wrong input format

    def test_predict_same(self):
        prediction_scores = self.tss.predict(self.data_X, self.data_X)
        self.assertTrue(prediction_scores == 1.0) #Same sentences

    def test_predict_dissame(self):
        prediction_scores = self.tss.predict(self.data_X, self.data_Y)
        self.assertTrue(prediction_scores < 1.0)  # Dissame sentences

if __name__ == '__main__':
    unittest.main()