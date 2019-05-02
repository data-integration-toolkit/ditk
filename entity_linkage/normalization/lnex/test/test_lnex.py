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
from entity_linkage.normalization.lnex.LNEX import LNExBase


class TestEntityNormalization(unittest.TestCase):

    def setUp(self):
        #find ditk_path from sys.path
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path:
                ditk_path = path
        #instantiate the implemented class
        self.lnex = LNExBase()
        #assume input file in same folder
        file_name = ditk_path+"/entity_linkage/normalization/lnex/test/sample_input.txt"
        self.test_set,self.eval_set = self.lnex.read_dataset(file_name)
        dummy = ""
        self.geo_info = self.lnex.train(dummy)


    def test_predict(self):
        #test if predicts returns at least a list with two elements 
        results = self.lnex.predict(self.geo_info, self.test_set)
        print(results)
        for line in results:
            self.assertTrue(line[0] != "" and line[1] != "")

    def test_evaluate(self):
        evaluations = self.lnex.evaluate(self.geo_info, self.eval_set)
        self.assertIsInstance(evaluations, dict)
        self.assertIn("precision", evaluations)
        self.assertIn("recall", evaluations)
        self.assertIn("f-score", evaluations)


if __name__ == '__main__':
    unittest.main()
