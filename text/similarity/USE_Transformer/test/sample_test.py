import unittest
import pandas as pd
from USE_Transformer import USE_Transformer_Similarity

class TestTextSimilarityMethods(unittest.TestCase):

    def setUp(self):
        self.tss = USE_Transformer_Similarity()
        self.data_X = ["This is a sample sentence to test text similarity"]
        self.data_Y = ["This is a different sentence to test text similarity"]
        local_module_path = '../model/moduleA' #the downloaded local module path
        remote_module_path = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
        self.tss.load_model(local_module_path)
        self.empty_list = []

    def test_predict_format(self):
        prediction_scores = self.tss.predict(self.data_X, self.data_Y)
        for score in prediction_scores:
            self.assertTrue(score >= 0 and score <= 1.0)
        with self.assertRaises(TypeError):
            self.tss.predict(self.empty_list) #Fails on wrong input format

    def test_predict_same(self):
        prediction_scores = self.tss.predict(self.data_X, self.data_X)
        for score in prediction_scores:
            self.assertTrue(score == 1.0) #Same sentences

    def test_predict_dissame(self):
        prediction_scores = self.tss.predict(self.data_X, self.data_Y)
        for score in prediction_scores:
            self.assertTrue(score < 1.0)  # Dissame sentences

if __name__ == '__main__':
    unittest.main()