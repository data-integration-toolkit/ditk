import unittest
import pandas as pd
from graph.completion.graph_completion import graph_completion


class TestGraphEmbeddingMethods(unittest.TestCase):

	def setUp(self, input_file):
		self.graph_embedding = GraphEmbedding() # initializes your Graph Embedding class
		self.input_file = input_file

	def test_read_dataset(self):
		train, validation, test = self.graph_embedding.read_dataset()
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(train, list) # assert non-empty list
		self.assertTrue(validation, list) # assert non-empty list
		self.assertTrue(test, list) # assert non-empty list

	def test_learn_embeddings(self):
		#Check output of the learned embedding
		output_vec = graph_embedding.main(self.input_file)
		embedding_vector = np.array(output_vec)
    	assertEquals(embedding_vector.shape[0],3)
    	assertEquals(embedding_vector.shape[1],300) #Example: output vec should be 3 x 300

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
