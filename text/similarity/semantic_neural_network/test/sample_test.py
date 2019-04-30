import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_neural_network import Semantic_Neural_Network

class TestTextSimilarityMethods(unittest.TestCase):

    def setUp(self):
        self.tss = Semantic_Neural_Network() #Your implementation of TextSemanticSimilarity
        self.data_X = "This is a sample sentence to test text similarity"
        self.data_Y = "This is a different sentence to test text similarity"
        self.empty_list = []

    def test_predict_format(self):
        prediction_score = self.tss.predict(self.data_X, self.data_Y)
        assertTrue(isFloat(prediction_score))
        assertTrue(prediction_score >= 0 and prediction_score <= 1.0)
        with self.assertRaises(ValueError):
            self.tss.predict(self.empty_list) #Fails on wrong input format

    def test_predict_similar(self):
        prediction_score = self.tss.predict(self.data_X, self.data_X)
        assertTrue(prediction_score == 1.0) #Same sentences

    def test_predict_dissimilar(self):
        prediction_score = self.tss.predict(self.data_X, self.data_Y)
        assertTrue(prediction_score < 1.0) #Dissimilar sentences

if __name__ == '__main__':
    unittest.main()