import unittest
import graph.embedding.HolE.skge.param
from graph.embedding.HolE import Holographic_Embedding
import os
import numpy as np

class TestGraphEmbeddingMethods(unittest.TestCase):

	def setUp(self):
		#self.graph_embedding = GraphEmbedding() # initialize your Blocking method

		self.graph_embedding = Holographic_Embedding.HolE_Embedding()
		self.fileName = ['/Users/boyuzhang/ditk/graph/embedding/HolE/tests/test_input/train.txt','/Users/boyuzhang/ditk/graph/embedding/HolE/tests/test_input/valid.txt','/Users/boyuzhang/ditk/graph/embedding/HolE/tests/test_input/test.txt']
		self.output_file = "output.txt"
		self.output= self.graph_embedding.main(self.fileName)
		self.train, self.validation, self.test, self.entities, self.relations = self.graph_embedding.read_dataset()
		self.model, self.ev_test = self.graph_embedding.return_parameter()

	def test_read_dataset(self):
		train, validation, test, entities, relations = self.graph_embedding.read_dataset()
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(train, list) # assert non-empty list
		self.assertTrue(validation, list) # assert non-empty list
		self.assertTrue(test, list) # assert non-empty list

	def test_learn_embeddings(self):
		# This fucntion could check on whether the embeddings were generated and if yes, then
		# it can check on whether the file exists

		self.assertEquals(len(self.entities) + len(self.relations), len(self.graph_embedding.model.E) + len(self.graph_embedding.model.R))
		self.assertEquals(len(self.graph_embedding.model.E[0]), 300)  # Example: output vec should be 3 x 300


	def test_evaluate(self):

		evaluations = self.graph_embedding.evaluate()
		# Evaluations could be a dictionary or a sequence of metrics names
		self.assertIsInstance(evaluations, dict)
		self.assertIn("MR", evaluations)
		self.assertIn("Hits", evaluations)

		mrr = evaluations["MR"]
		hits = evaluations['Hits']
		self.assertIsInstance(mrr, float)
		if hits is not None:
			for hit in hits:
				self.assertIsInstance(hits[hit], float)

if __name__ == '__main__':
    unittest.main()