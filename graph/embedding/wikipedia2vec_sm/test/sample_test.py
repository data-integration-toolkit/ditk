import unittest
import pandas as pd
import numpy as np
#from graph.completion.graph_completion import graph_completion
from Wikipedia2vec import Wikipedia2vec


class TestGraphEmbeddingMethods(unittest.TestCase):
	def setUp(self):
		#self.graph_embedding = GraphEmbedding() # initializes your Graph Embedding class
		self.graph_embedding = Wikipedia2vec()
		self.input_file = 'sample_input.txt'
	
	def test_read_dataset(self):
		train, validation, test = self.graph_embedding.read_dataset()
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(train, list) # assert non-empty list
		self.assertTrue(validation, list) # assert non-empty list
		self.assertTrue(test, list) # assert non-empty list
	
	def test_learn_embeddings(self):
		#Check output of the learned embedding
		#output_vec = graph_embedding.main(self.input_file)
		output_vec = self.graph_embedding.learn_embeddings('output.db','output_dic','final_output_text')
		embedding_vector = np.array(output_vec)
		self.assertEquals(embedding_vector.shape[0],10039)
		self.assertEquals(embedding_vector.shape[1],2) #Example: output vec should be 3 x 300
	
	def test_evaluate(self):
		evaluations = self.graph_embedding.evaluate()
		f1 = self.graph_embedding.evaluate()
		self.assertIsInstance(f1, float)
		


if __name__ == '__main__':
    unittest.main()
