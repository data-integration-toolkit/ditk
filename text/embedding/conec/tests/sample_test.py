import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conec import conec

class TestTextEmbeddingMethods(unittest.TestCase):

    def setUp(self):
        self.te = conec() #Your implementation of TextEmbedding
        self.input_token1 = "student"
        self.input_token2 = "students"
        self.input_sent1 = "I am a girl"
        self.input_sent2 = "I am a woman"

    def test_predict_embedding(self):
        embedding_vector = self.te.predict_embedding(self.input_token1)
        embedding_vector = np.array(embedding_vector)
        self.assertEqual(embedding_vector.shape[0],200)
        # self.assertEqual(embedding_vector.shape[1],200) #Example: output vec should be 1 x 200

    def test_predict_sent_embedding(self):
        embedding_vector = self.te.predict_sent_embedding(self.input_sent1)
        embedding_vector = np.array(embedding_vector)
        self.assertEqual(embedding_vector.shape[0],200)
        # self.assertEqual(embedding_vector.shape[1],1)

    def test_predict_similarity(self):
        prediction_score = self.te.predict_similarity(self.input_token1, self.input_token2)
        self.assertTrue(float(prediction_score))
        self.assertTrue(prediction_score >= 0 and prediction_score <= 1.0)
    
    def test_predict_sent_similarity(self):
        prediction_score = self.te.predict_sent_similarity(self.input_sent1, self.input_sent2)
        self.assertTrue(float(prediction_score))
        self.assertTrue(prediction_score >= 0 and prediction_score <= 1.0)

if __name__ == '__main__':
    unittest.main()
