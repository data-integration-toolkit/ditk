import unittest
import numpy as np

from ntee.model_reader import ModelReader

class TestTextEmbeddingMethods(unittest.TestCase):

    def setUp(self):

        self.te = ModelReader('/Users/ashiralam/Downloads/ntee-master/ntee/ntee_300_sentence.joblib')

        self.input_tokens = (u"sample")

    def test_predict_embedding(self):
        x = self.te.get_word_vector(self.input_tokens)
        embedding_vector = np.array(x)

        print embedding_vector
        self.assertEquals(embedding_vector.shape[0],300)


if __name__ == '__main__':
    unittest.main()
