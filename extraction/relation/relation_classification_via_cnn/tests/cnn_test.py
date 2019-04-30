# uses python 3.5 and tensorflow 1.4.0

import unittest
from cnn_model import CNNModel
import os

class RelationExtractionTest(unittest.TestCase):
    def setUp(self):
        self.model = CNNModel()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_train_file = dir_path+'\\re_input_semeval_train.txt'
        self.input_test_file = dir_path+'\\re_input_semeval_test.txt'
        self.data_folder = './data'
        self.model_folder = './saved_models'
        self.output_file = dir_path+'\\re_output_semeval.txt'
    
    def test_read_dataset(self):
        self.model.read_dataset(self.input_train_file)

        # assert output data folder exists
        assert os.path.isdir(self.data_folder)
        assert os.path.exists(self.data_folder)
    
    def test_train(self):
        self.model.train(self.data_folder)

        # assert trained model exists
        assert os.path.isdir(self.model_folder)
        assert os.path.exists(self.model_folder)

    def test_predict(self):
        self.model.predict(self.input_test_file)

        with open(self.output_file, 'r') as of:
            output1 = of.readline()

        # assert 5 attributes, separated by tab
        self.assertEqual(len(output1.split("\t")), 5)
    
    def test_evaluate(self):
        precision, recall, f1 = self.model.evaluate(self.data_folder)

        # assert float
        self.assertTrue(precision, float)
        self.assertTrue(recall, float)
        self.assertTrue(f1, float)

        print("\nEvaluation")
        print("Precision: "+str(precision))
        print("Recall: "+str(recall))
        print("F1: "+str(f1))


if __name__ == '__main__':
    unittest.main()