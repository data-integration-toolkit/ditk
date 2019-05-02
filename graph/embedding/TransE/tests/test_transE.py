# This is a main file that evoke entire codes

import sys
import os
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, dir_path + "/src")

#from main import main
#dir_path = os.path.dirname(os.path.realpath(__file__))
#print (dir_path)
import unittest
import tensorflow as tf
from graph_embedding_transE import MyTransE

class TestGraphEmbeddingMethods(unittest.TestCase):
    def setUp(self):
            #self.graph_embedding = GraphEmbedding() # initialize your Blocking method
            self.graph_embedding = MyTransE()
            self.input_file = './data/YAGO/'
            self.dimension = 300
            self.marginal_value = 1.0
            self.batch_size = 4800
            self.max_epoch = 3

    def test_read_dataset(self):
            train, validation, test = self.graph_embedding.read_dataset(self.input_file)
            # If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
            self.assertTrue(train, list) # assert non-empty list
            self.assertTrue(validation, list) # assert non-empty list
            self.assertTrue(test, list) # assert non-empty list

    def test_learn_embeddings(self):
            self.graph_embedding.read_dataset(self.input_file)
            # This fucntion could check on whether the embeddings were generated and if yes, then
            # it can check on whether the file exists
            embedding_vector, n_entity, n_relation = self.graph_embedding.learn_embeddings(self.dimension, self.marginal_value, self.batch_size, self.max_epoch)

            self.graph_embedding.save_model('./output/')

            #assert os.path.exists("./embedded_entity.out")
            #assert os.path.exists("./embedded_relations.out")

            self.assertEqual(embedding_vector.shape[0], (n_entity + n_relation))
            self.assertEqual(embedding_vector.shape[1],300) #Example: output vec should be 3 x 300

    def test_evaluate(self):
            tf.reset_default_graph()
            self.graph_embedding.read_dataset(self.input_file)
            self.graph_embedding.learn_embeddings(self.dimension, self.marginal_value, self.batch_size, self.max_epoch)
            evaluations = self.graph_embedding.evaluate()
            # Evaluations could be a dictionary or a sequence of metrics names

            self.assertIsInstance(evaluations, dict)
            self.assertIn("MRR", evaluations)
            self.assertIn("Hits", evaluations)

            #print('-----Average-----')
            #print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(evaluations['MRR'],evaluations['Hits']))

            file = open("./output/Evaluation.txt","w")
            file.write('-----Average-----')
            file.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(evaluations['MRR'],evaluations['Hits']))
            file.close()

if __name__ == '__main__':
    unittest.main()
    """
    # initial TransE
    TransE = TransE()

    # Set the data to use
    # Select from ["FB15", "WN18", "test case"]
    TransE.read_dataset("WN18")
    print(TransE.dataset_name)

    TransE.learn_embeddings()
    print(TransE.graph_embedded_entity.shape)
    print(TransE.graph_embedded_relations.shape)

    TransE.save_model('./')
    """
