import unittest
import pandas as pd
import networkx as nx
import numpy as np
from graph.embedding.graph_embedding import graph_embedding


class TestGraphEmbeddingMethods(unittest.TestCase):


	def test_read_dataset(self):
		graph = self.graph_embedding.read_dataset()
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(graph, nx.Graph)

	def test_learn_embeddings(self):
		#Check output of the learned embedding
		output_vec = graph_embedding.main(self.input_file)
		embedding_vector = np.array(output_vec)
    	assertEquals(embedding_vector.shape[1],128)
	def test_evaluate(self):
		evaluations = self.graph_embedding.evaluate()
		# Evaluations could be a dictionary or a sequence of metrics names
		self.assertIn("cosine similarity", evaluations)
		self.assertIsInstance(cs, float)


if __name__ == '__main__':
    unittest.main()
