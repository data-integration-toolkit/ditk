import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import pickle
from rescal_model import RESCALModel

class RESCALTest(unittest.TestCase):
    def setUp(self):
        self.model = RESCALModel()
        dir_path = os.path.join( os.path.dirname( __file__ ), '..')+'\\benchmark1'
        self.input_file = dir_path+'\\rescal_input_alyawarradata.mat'
        self.key_tensor = 'Rs'
        self.output_dir = dir_path
    
    def test_read_dataset(self):
        X = self.model.read_dataset(self.input_file, self.key_tensor)
        
        # assert list
        self.assertTrue(X, list)

    def test_factorize(self):
        X = self.model.read_dataset(self.input_file, self.key_tensor)
        m = len(X)
        n, _ = X[0].shape
        r = 50
        A, R = self.model.factorize(r)

        # assert A is numpy array
        self.assertTrue(A.tolist(), list)
        # assert shape of A is n * r
        self.assertEqual(A.shape, (n, r))
        # assert R is list
        self.assertTrue(R, list)
        # assert shape of each element of R is r * r
        self.assertEqual(R[0].shape, (r, r))

    def test_save_model(self):
        # remove saved matrices A and R txt files if they exist
        if os.path.exists(self.output_dir+'\\rescal_output_A.txt'):
            os.remove(self.output_dir+'\\rescal_output_A.txt')
        if os.path.exists(self.output_dir+'\\rescal_output_R.txt'):
            os.remove(self.output_dir+'\\rescal_output_R.txt')

        self.model.read_dataset(self.input_file, self.key_tensor)
        self.model.factorize(50)
        self.model.save_model(self.output_dir)

        # assert matrices A and R are outputted to txt files
        assert os.path.exists(self.output_dir+'\\rescal_output_A.txt')
        assert os.path.exists(self.output_dir+'\\rescal_output_R.txt')

    def test_load_model(self):
        self.model.read_dataset(self.input_file, self.key_tensor)
        A, R = self.model.factorize(50)
        self.model.save_model(self.output_dir)

        # assert matrices A and R outputted to txt files are equivalent to that of the original ones
        output_A = np.loadtxt(self.output_dir+'\\rescal_output_A.txt', delimiter=',')
        self.assertEqual(A.tolist(), output_A.tolist())

        with open(self.output_dir+"\\rescal_output_R.txt", "rb") as rb:
            output_R = pickle.load(rb)

            for r1,r2 in zip(R, output_R):
                self.assertEqual(r1.tolist(), r2.tolist())
        
    def test_evaluate(self):
        self.model.read_dataset(self.input_file, self.key_tensor)
        mean_train, std_train, mean_test, std_test = self.model.evaluate()

        # assert float
        self.assertTrue(mean_train, float)
        self.assertTrue(std_train, float)
        self.assertTrue(mean_test, float)
        self.assertTrue(std_test, float)

        print("\nEvaluation")
        print("PR AUC training/test mean: %.3f/%.3f" %(mean_train, mean_test))
        print("PR AUC training/test standard deviation: %.3f/%.3f" %(std_train, std_test))

if __name__ == '__main__':
    unittest.main()