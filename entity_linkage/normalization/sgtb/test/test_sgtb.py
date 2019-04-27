import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import unittest
from structured_gradient_tree_boosting import StructuredGradientTreeBoosting


class TestEntityNormalization(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        self.sgtb = StructuredGradientTreeBoosting()
        input_file = './test/sample_input.json'
        path_to_model = './model/finalized_model.sav'
        split_ratio = (0.7, 0.15, 0.15)
        self.model = self.sgtb.load_model(path_to_model)
        _, self.test, _ = self.sgtb.read_dataset(input_file, split_ratio)


    def test_predict(self):
        #test if predicts returns at least a list with two elements 
        results = self.sgtb.predict(self.model, self.test)
        for line in results:
            assertTrue(line[0] != "")

    def test_evaluate(self):
        evaluations = self.sgtb.evaluate(self.model, self.test)
        self.assertIsInstance(evaluations, float)


if __name__ == '__main__':
    unittest.main()
