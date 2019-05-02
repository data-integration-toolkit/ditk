import unittest
import pandas as pd
import numpy as np
from graph.completion.graph_completion import graph_completion


class TestGraphCompletionMethods(unittest.TestCase):

    
	def test_read_dataset(self):
		adj, features = self.graph_completion.read_dataset()
		# You need to make sure that the output format of
		# the read_dataset() function for any given input remains the same
		self.assertTrue(adj, np.array) # assert non-empty list
		self.assertTrue(features, np.array) # assert non-empty list
	    
	def test_predict(self):
		predictions = self.graph_completion.predict()
		# evaluate whether predictions follow a common format such as:
		# each tuple in the output likely will follow format: (subject_entity, relation, object_entity)
		self.assertTrue(predictions, np.array)  # assert non-empty list

	def test_evaluate(self):
		evaluations = self.graph_completion.evaluate()
		# Make sure that the returned metrics are inside a dictionary and the required keys exist
		self.assertIn("AUC", evaluations)
		self.assertIn("AP", evaluations)



if __name__ == '__main__':
    unittest.main()
