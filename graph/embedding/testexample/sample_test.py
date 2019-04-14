import unittest
import pandas as pd
from graph.completion.graph_completion import graph_completion


class TestGraphEmbeddingMethods(unittest.TestCase):

	def setUp(self, input_file):
		self.graph_embedding = GraphEmbedding() # initialize your Blocking method
		self.input_file = input_file

	def test_read_dataset(self):
		train, validation, test = self.graph_embedding.read_dataset()
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(train, list) # assert non-empty list
		self.assertTrue(validation, list) # assert non-empty list
		self.assertTrue(test, list) # assert non-empty list

	def test_learn_embeddings(self):
		# This fucntion could check on whether the embeddings were generated and if yes, then
		# it can check on whether the file exists
		pass

	def test_evaluate(self):
		evaluations = self.graph_embedding.evaluate()
		# Evaluations could be a dictionary or a sequence of metrics names
		self.assertIsInstance(evaluations, dict)
		self.assertIn("f1", evaluations)
		self.assertIn("MRR", evaluations)
		self.assertIn("Hits", evaluations)

		f1, mrr, hits = self.graph_embedding.evaluate()
		self.assertIsInstance(f1, float)
		self.assertIsInstance(mrr, float)
		if hits is not None:
			self.assertIsInstance(hits, float)


if __name__ == '__main__':
    unittest.main()