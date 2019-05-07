import unittest
import pandas as pd
import ProjE
import main
import numpy as np

class TestGraphEmbeddingMethods(unittest.TestCase):

	def setUp(self):
		self.graph_embedding = ProjE.ProjE() # initializes your Graph Embedding class
		self.input_file = './sample_input_yago/'

	def test(self):
		args, train, validation, test = self.graph_embedding.read_dataset(self.input_file)
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
		self.assertTrue(train, list) # assert non-empty list
		self.assertTrue(validation, list) # assert non-empty list
		self.assertTrue(test, list) # assert non-empty list
		#Check embeddings
		train_hrt_input, train_hrt_weight, file, train_trh_input, train_trh_weight, train_loss, train_op, ent, rel = self.graph_embedding.learn_embeddings(data = self.input_file, argDict = args)
		embedding_vector_ent = np.array(ent)
		embedding_vector_rel = np.array(rel)
		self.assertEqual(embedding_vector_ent.shape[0], 10623) #10623 entities in YAGO
		self.assertEqual(embedding_vector_ent.shape[1], 200) #Example: output vec should be 1x200
		self.assertEqual(embedding_vector_rel.shape[0], 10) #10 relations in YAGO
		self.assertEqual(embedding_vector_rel.shape[1], 200)

if __name__ == '__main__':
    unittest.main()
