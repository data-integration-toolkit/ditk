# This is a main file that evoke entire codes
import os
import sys
import unittest
import numpy as np

#from text_embedding_similarity import SimilarityEncoding
#from fit_predict_categorical_encoding import fit_predict_categorical_encoding

# Import original transE source codes
#import sys
#import os
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, dir_path + "../src")

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.abspath("../"), "src"))

# from os.path import dirname
#sys.path.append(dirname('../src'))
# sys.path.append(os.path.join(os.path.abspath("../"), "src"))
os.chdir("../src")
#dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)

from text_embedding_similarity import SimilarityEncoding

class TestTextEmbeddingMethods(unittest.TestCase):
    def setUp(self):
        self.te = SimilarityEncoding(300)
        #self.te.read_Dataset('./input.txt')
        #self.te.train()
        #self.input_tokens = ["sample"]

    def read_dataset(self):
        #read in training data/data that need to be embeded
        self.te.read_Dataset('../tests/test_data/input_train.txt')
        print(self.te.input_data)

    def train(self):
        #Embeded the input data
        self.te.read_Dataset('../tests/test_data/input_train.txt')
        output = self.te.train()
        unWords = np.unique(self.te.input_data)
        #print(output)

        index = 0
        file = open("../tests/test_data/output_embedded_train.txt","w")
        for x in output.tolist():
            file.write(str(unWords[index]) + ' ' )
            file.write(str(x))
            file.write("\n")
            index += 1
        file.close()

    def predict_embedding(self):
        #Predict the inputs by comparing it to training data
        self.te.read_Dataset('../tests/test_data/input_train.txt')
        self.te.train()
        output = self.te.predict_embedding('../tests/test_data/input_predict.txt')
        unWords = np.unique(self.te.predict_data)
        #print(output.shape)
        output = output.tolist()

        index = 0
        file = open("../tests/test_data/output_embedded_predict.txt","w")
        for x in output:
            file.write(str(unWords[index]) + ' ' )
            file.write(str(x))
            file.write("\n")
            index += 1
        file.close()

    def predict_similarity(self):
        #Given 2 input, embeded them with input data, them compare them using pearson similarity
        self.te.read_Dataset('../tests/test_data/input_train.txt')
        self.te.train()
        pearson = self.te.predict_similarity('awsl', 'nbcs')

        print(pearson)

    def test_evaluate(self):
        self.te.read_Dataset('../data/Semi2017/input.txt')
        self.te.train()
        result = self.te.evaluate("Semi2017",'../data/Semi2017/sts-test-2017-small.csv')
        print(result)

if __name__ == '__main__':
    unittest.main()
