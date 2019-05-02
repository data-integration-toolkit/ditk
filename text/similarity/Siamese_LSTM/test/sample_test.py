import unittest
import pandas as pd
from text.similarity.Siamese_LSTM.src.Siamese_LSTM_Similarity import Siamese_LSTM_Similarity
from decimal import Decimal

class TestTextSimilarityMethods(unittest.TestCase):

    def setUp(self):
        self.tss = Siamese_LSTM_Similarity()
        self.data_X = "This is a sample sentence to test text similarity"
        self.data_Y = "This is a different sentence to test text similarity"
        self.tss.load_model('../model/bestsem.p')
        self.empty_list = []

    def test_predict_format(self):
        prediction_score = self.tss.predict(self.data_X, self.data_Y)
        self.assertTrue(prediction_score >= 0 and prediction_score <= 1.0)
        with self.assertRaises(TypeError):
            self.tss.predict(self.empty_list) #Fails on wrong input format

    def test_predict_same(self):
        prediction_score = self.tss.predict(self.data_X, self.data_X)
        prediction_score = Decimal(prediction_score).quantize(Decimal('0.00'))
        self.assertTrue(prediction_score == 1.0) #Same sentences

    def test_predict_dissame(self):
        prediction_score = self.tss.predict(self.data_X, self.data_Y)
        prediction_score = Decimal(prediction_score).quantize(Decimal('0.00'))
        self.assertTrue(prediction_score < 1.0)  # Dissame sentences

if __name__ == '__main__':
    unittest.main()