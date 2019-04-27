import sys, os

# set path to ditk root
ditk_path = os.path.abspath(os.getcwd())
if ditk_path not in sys.path:
    sys.path.append(ditk_path)

data_dir = os.path.join(ditk_path, 'graph/completion/longae/test/')
input_files = list(map(lambda x: os.path.join(data_dir, x), ['test_input_x.txt', 'test_input_y.txt', 'test_input_graph.txt']))
output_files = list(map(lambda x: os.path.join(data_dir, x), ['test_output_y.txt', 'test_output_graph.txt']))

# adjust hparam for testing
from graph.completion.longae.hparams import hparams as hp
hp.training_size = 5
hp.dev_size = 2
import unittest
import pandas as pd

from graph.completion.longae.longae import longae

class TestGraphCompletionMethods(unittest.TestCase):

    def setUp(self):
        self.graph_completion = longae() # initialize your Graph Completion class
        self.input_files = input_files

    def test_read_dataset(self):
        train, test, dev = self.graph_completion.read_dataset(self.input_files)
        # You need to make sure that the output format of
        # the read_dataset() function for any given input remains the same
        self.assertTrue(train, list) # assert non-empty list
        self.assertTrue(test, list) # assert non-empty list
        self.assertTrue(dev, list) # assert non-empty list

    def test_predict(self):
        _, test, _ = self.graph_completion.read_dataset(self.input_files)
        predictions = self.graph_completion.predict(test)
        # evaluate whether predictions follow a common format such as:
        # each tuple in the output likely will follow format: [0, 1, 1, 0, ...]
        self.assertTrue(predictions['lp_scores'], list)  # assert non-empty list
        self.assertTrue(predictions['nc_scores'], list)  # assert non-empty list

    def test_evaluate(self):
        from sklearn.metrics import roc_auc_score as auc_score
        from sklearn.metrics import average_precision_score as ap_score

        metrics = {
            "auc_score": auc_score,
            "ap_score": ap_score
        }

        _, _, test = self.graph_completion.read_dataset(self.input_files)
        predictions = self.graph_completion.predict(test)
        evaluations = self.graph_completion.evaluate(test, metrics, predictions)

        # Make sure that the returned metrics are inside a dictionary and the required keys exist
        self.assertIsInstance(evaluations, dict)
        self.assertIn("AUC", evaluations)
        self.assertIn("AP", evaluations)
        self.assertIn("ACC", evaluations)

if __name__ == '__main__':
    unittest.main()