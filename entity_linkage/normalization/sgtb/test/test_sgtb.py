import os
import sys


# running test_sgtb.py in test folder
if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import unittest
from entity_linkage.normalization.sgtb.structured_gradient_tree_boosting import StructuredGradientTreeBoosting


class TestEntityNormalization(unittest.TestCase):

    def setUp(self):
        #find ditk_path from sys.path
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path:
                ditk_path = path
        #instantiate the implemented class
        self.sgtb = StructuredGradientTreeBoosting()
        #assume input file in same folder and model stored at model
        input_file = ditk_path+'/entity_linkage/normalization/sgtb/test/sample_input.json'
        path_to_model = ditk_path+'/entity_linkage/normalization/sgtb/model/finalized_model.sav'
        split_ratio = (0.7, 0.15, 0.15)
        self.model = self.sgtb.load_model(path_to_model)
        _, self.test, _ = self.sgtb.read_dataset(input_file, split_ratio)


    def test_predict(self):
        #test if predicts returns at least a list with two elements 
        results = self.sgtb.predict(self.model, self.test)
        for line in results:
            self.assertTrue(line[0] != "" and line[1] != "")

    def test_evaluate(self):
        evaluations = self.sgtb.evaluate(self.model, self.test)
        self.assertIsInstance(evaluations, float)


if __name__ == '__main__':
    unittest.main()
