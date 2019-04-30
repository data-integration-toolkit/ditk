import unittest
import numpy as np
import os

from fasttext import FastText

dirpath = os.getcwd()
datapath = dirpath + "/data/sick.xls"
modelpath = dirpath + "/models/log_sick"
class TextEmbeddingMethods(unittest.TestCase):

    def setUp(self):
        self.te = FastText(input_corpus_path=datapath,modelpath=modelpath) #Your implementation of TextEmbedding
        self.input_tokens = ["man"]

    def test_predict_embedding(self):
        model, dict_map = self.te.load_model(modelpath)
        embedding_vector = self.te.predict_embedding(self.input_tokens, model, dict_map)
        embedding_vector = np.array(embedding_vector)
        self.assertEqual(embedding_vector.shape[0],200)
        #self.assertEqual(embedding_vector.shape[1],200) #Example: output vec should be 1 x 300

if __name__ == '__main__':
    unittest.main()
