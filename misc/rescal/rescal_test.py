import unittest
from rescal_model import RESCALModel
import os
import numpy as np
import pickle

class RESCALTest(unittest.TestCase):
    def setUp(self):
        self.model = RESCALModel()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_file = dir_path+'\\rescal_input_alyawarradata.mat'
        self.key_tensor = 'Rs'
        self.output_A_file = dir_path+'\\rescal_output_A.txt'
        self.output_R_file = dir_path+"\\rescal_output_R.txt"
    
    def test_read_dataset(self):
        X = self.model.read_dataset(self.input_file, self.key_tensor)
        
        # assert list
        self.assertTrue(X, list)

    def test_factorize(self):
        self.model.read_dataset(self.input_file, self.key_tensor)
        A, R = self.model.factorize(50)

        # assert numpy array
        self.assertTrue(A.tolist(), list)
        # assert list
        self.assertTrue(R, list)

    def test_outputs(self):
        self.model.read_dataset(self.input_file, self.key_tensor)
        A, R = self.model.factorize(50)

        # assert matrices A and R outputted to txt files are equivalent to that of the original ones
        output_A = np.loadtxt(self.output_A_file, delimiter=',')
        self.assertEqual(A.tolist(), output_A.tolist())

        with open(self.output_R_file, "rb") as rb:
            output_R = pickle.load(rb)

            for r1,r2 in zip(R, output_R):
                self.assertEqual(r1.tolist(), r2.tolist())

    def test_evaluate(self):
        self.model.read_dataset(self.input_file, self.key_tensor)
        self.model.factorize(50)
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