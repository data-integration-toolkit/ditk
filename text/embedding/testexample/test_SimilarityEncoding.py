# This is a main file that evoke entire codes

import os
import sys
import unittest
import numpy as np

from text_embedding_similarity import SimilarityEncoding
from fit_predict_categorical_encoding import fit_predict_categorical_encoding

class TestTextEmbeddingMethods(unittest.TestCase):
    def setUp(self):
        self.te = SimilarityEncoding()
        self.te.read_Dataset('./input.txt')
        self.te.train()
        #self.input_tokens = ["sample"]

    def test_predict_embedding(self):

        file = open("text_embedding_output.txt","w")

        embedding_vector = self.te.predict_embedding("Man")
        #embedding_vector = np.array(vec
        self.assertEquals(embedding_vector.shape[0],1)
        self.assertEquals(embedding_vector.shape[1],200)

        file.write("Man\n")
        for x in embedding_vector.tolist():
            file.write(str(x))
        file.write("\n")

        embedding_vector = self.te.predict_embedding("Woman")
        self.assertEquals(embedding_vector.shape[0],1)
        self.assertEquals(embedding_vector.shape[1],200)

        file.write("Woman\n")
        for x in embedding_vector.tolist():
            file.write(str(x))
        file.write("\n")

        embedding_vector = self.te.predict_embedding("King")
        self.assertEquals(embedding_vector.shape[0],1)
        self.assertEquals(embedding_vector.shape[1],200)

        file.write("King\n")
        for x in embedding_vector.tolist():
            file.write(str(x))
        file.write("\n")

        embedding_vector = self.te.predict_embedding("Queen")
        self.assertEquals(embedding_vector.shape[0],1)
        self.assertEquals(embedding_vector.shape[1],200)

        file.write("Queen\n")
        for x in embedding_vector.tolist():
            file.write(str(x))
        file.write("\n")

        file.close()

if __name__ == '__main__':
    unittest.main()
