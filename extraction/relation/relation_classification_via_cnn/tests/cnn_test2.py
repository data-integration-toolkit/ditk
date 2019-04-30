# Note: run cnn_test1.py before cnn_test2.py

import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cnn_model import CNNModel

class RelationExtractionTest(unittest.TestCase):
    def setUp(self):
        self.model = CNNModel()
        dir_path = os.path.dirname(__file__)
        dir_path = dir_path.replace(os.sep, '/')
        self.input_dir = dir_path+'/data'
        self.model_dir = dir_path+'/model'
    
    def test_predict_and_evaluate(self):
        self.model.read_dataset(self.input_dir)
        output_file_path = self.model.predict(self.input_dir, trained_model=self.model_dir)

        with open(output_file_path, 'r') as of:
            output1 = of.readline()

        # assert 5 attributes, separated by tab
        self.assertEqual(len(output1.split("\t")), 5)
        
        precision, recall, f1 = self.model.evaluate(self.input_dir)

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