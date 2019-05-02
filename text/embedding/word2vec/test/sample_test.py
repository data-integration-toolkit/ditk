import unittest
import numpy as np

from word2vec import Word2Vec_Util 

class TestTextEmbeddingMethods(unittest.TestCase):

    def setUp(self):
        self.te = Word2Vec_Util() #Your implementation of TextEmbedding
        self.input_tokens = ["dogs"]

    def test_predict_embedding(self):
        embedding_vector = self.te.predict_embedding(self.input_tokens)
        self.assertTrue(len(embedding_vector)>0) # Embedding Generation Verification
        embedding_vector = np.array(embedding_vector)
        self.assertEqual(embedding_vector.shape[0],200) # Embedding size verification

if __name__ == '__main__':
    unittest.main()