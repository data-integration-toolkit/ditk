import os
import sys

module_path = os.path.abspath(os.path.join('../../../..'))
if module_path not in sys.path:
  sys.path.append(module_path)

import unittest
import pandas as pd
from graph.completion.deepPath.deep_path import DeepPath
from graph.completion.deepPath.main import main
class TestGraphCompletionMethods(unittest.TestCase):
    def setUp(self):
        self.deepPath = DeepPath()
        self.input_file = "./sample_input.txt"
        self.output_file = main(self.input_file)
  
    def test_read_dataset(self) :
        return_dict = self.deepPath.read_dataset(self.input_file)
        self.assertTrue(return_dict, dict)
        self.assertTrue(return_dict.values(), list)
        self.assertTrue(type(list(return_dict.values())[0]), str)

    def test_evaluate(self):
        return_dict = self.deepPath.read_dataset(self.input_file)
        self.deepPath.train(return_dict)
        self.deepPath.predict(return_dict)
        evaluation = self.deepPath.evaluate(return_dict)
        self.assertIn("MAP", evaluation)

if __name__ == '__main__':
    unittest.main()

