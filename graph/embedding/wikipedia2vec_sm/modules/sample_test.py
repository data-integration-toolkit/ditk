import unittest
import pandas as pd
import numpy as np
#from graph.completion.graph_completion import graph_completion
from wikipedia2vec_sm import Wikipedia2vec


class TestGraphEmbeddingMethods(unittest.TestCase):
	def setUp(self):
		#self.graph_embedding = GraphEmbedding() # initializes your Graph Embedding class
		self.graph_embedding = Wikipedia2vec()
		self.input_file = 'yago.txt'
	
	def test_read_dataset(self):
		train, validation, test = self.graph_embedding.read_dataset(self.input_file)
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(train, list) # assert non-empty list
		self.assertTrue(validation, list) # assert non-empty list
		self.assertTrue(test, list) # assert non-empty list
	
	def test_learn_embeddings(self):
		output_vec, out_file = self.graph_embedding.load_model(self.input_file)
		embedding_vector = np.array(output_vec)
		self.assertEquals(embedding_vector.shape[0],1474)
		self.assertEquals(embedding_vector.shape[1],2) #Example: output vec should be 3 x 300

	def test_evaluate(self):
		f1 = self.graph_embedding.evaluate("embeddings.txt")
		self.assertIsInstance(f1, float)
		


if __name__ == '__main__':
    unittest.main()
