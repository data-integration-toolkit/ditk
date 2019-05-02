import unittest
import pandas as pd
import os,sys
sys.path.append("..")
import cnn_sts
from cnn_sts import STSTask, Embedder
from gensim.models import KeyedVectors


class TestTextSimilarityMethods(unittest.TestCase):

    def setUp(self):
        #download the vector file in this repo
        vecfile = ''
        vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)
        self.tss = STSTask() #Your implementation of TextSemanticSimilarity
        self.data_X = "This is a sample sentence to test text similarity"
        self.data_Y = "This is a different sentence to test text similarity"
        self.empty_list = [] 
        self.embed = Embedder(dictname, self.c['wordvectdim'])
        self.model = self.tss.load_model('cnn_model.h5')

    def test_predict_format(self):
        prediction_score = round(self.tss.predict(self.data_X, self.data_Y)[0],3)/5.0
        self.assertTrue(prediction_score >= 0 and prediction_score <= 1.0)
        # with self.assertRaises(ValueError):
        #     self.tss.predict_score(self.empty_list) #Fails on wrong input format

    def test_predict_similar(self):    
        prediction_score = round(self.tss.predict(self.data_X, self.data_X)[0],3)/5.0
        self.assertTrue(prediction_score >0.9) #Same sentences

    def test_predict_dissimilar(self):
        prediction_score = round(self.tss.predict(self.data_X, self.data_Y)[0],3)/5.0
        self.assertTrue(prediction_score < 1.0) #Dissimilar sentences

if __name__ == '__main__':
    unittest.main()