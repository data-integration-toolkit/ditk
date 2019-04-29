import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import unittest
import pandas as pd
from graph.completion.convKB.conv_kb import ConvKB
from graph.completion.convKB import main


class TestGraphCompletionMethods(unittest.TestCase):

    def setUp(self):
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path and not "graph" in path:
                ditk_path = path
        self.convKB = ConvKB() # initialize your specific Graph Completion class
        self.input_file = ditk_path+"/graph/completion/convKB/test/sample_input.txt"
        self.output_file = main.main(self.input_file)

    def test_read_dataset(self):
        train, dev, test = self.convKB.read_dataset(self.input_file)
        # You need to make sure that the output format of
        # the read_dataset() function for any given input remains the same
        self.assertTrue(train, list) # assert non-empty list
        self.assertTrue(dev, list) # assert non-empty list
        self.assertTrue(test, list) # assert non-empty list

    def test_predict(self):
        options = {}
        options["split_ratio"] = (0.7, 0.1, 0.2)
        _, _, test = self.convKB.read_dataset(self.input_file, options)
        predictions = self.convKB.predict(test)
        # evaluate whether predictions follow a common format such as:
        # each tuple in the output likely will follow format: (subject_entity, relation, object_entity)
        self.assertTrue(predictions, list)  # assert non-empty list

    def test_evaluate(self):
        options = {}
        options["split_ratio"] = (0.7, 0.1, 0.2)
        _, _, test = self.convKB.read_dataset(self.input_file, options)
        evaluations = self.convKB.evaluate(test)
        # Make sure that the returned metrics are inside a dictionary and the required keys exist
        self.assertIsInstance(evaluations, dict)
        self.assertIn("MRR", evaluations)
        self.assertIn("hit@10", evaluations)


if __name__ == '__main__':
    unittest.main()