import unittest
import numpy as np

from similarity import TextEmbedding

class TestTextEmbeddingMethods(unittest.TestCase):

	def setUp(self):
        self.te = TextEmbedding() #Your implementation of TextEmbedding
        self.input_tokens = ["sample"]

    def test_predict_embedding(self):
    	embedding_vector = te.predict_embedding(input_tokens)
    	embedding_vector = np.array(vec)
    	assertEquals(embedding_vector.shape[0],1)
    	assertEquals(embedding_vector.shape[1],300) #Example: output vec should be 1 x 300

if __name__ == '__main__':
    unittest.main()
