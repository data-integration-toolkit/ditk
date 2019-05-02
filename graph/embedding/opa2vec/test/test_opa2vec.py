import os
import sys
import numpy as np


if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import unittest
from opa2vec import OPA2VEC


class TestGraphEmbeddingMethods(unittest.TestCase):

    def setUp(self):
        #find ditk_path from sys.path
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path:
                ditk_path = path
        #instantiate the implemented class
        self.opa = OPA2VEC()
        #assume input file in same folder
        file_name = []
        onto = ditk_path+"/graph/embedding/opa2vec/test/graph_embedding_input1.owl"
        association = ditk_path+"/graph/embedding/opa2vec/test/graph_embedding_input2.txt"
        output_file = ditk_path+"/graph/embedding/opa2vec/test/sample_output.txt"
        file_name.append(onto)
        file_name.append(association)
        file_name.append(output_file)
        self.input_file = file_name
        #self.output_vec = []
        
    def test_read_dataset(self):
        ontology,association = self.opa.read_dataset(self.input_file)
		# If possible check if the read_dataset() function returns data of similar format (e.g. vectors of any size, lists of lists, etc..)
        self.assertTrue(ontology, list) # assert non-empty list
        self.assertTrue(association, list) # assert non-empty list
		


    def test_learn_embeddings(self):
		#Check output of the learned embedding
        output_vec = self.opa.learn_embeddings(self.input_file)
        embedding_vector = np.array(output_vec)
        self.assertEqual(embedding_vector.shape[0],296)
        self.assertEqual(embedding_vector.shape[1],200) #Example: output vec should be 296 x 200

    def test_evaluate(self):
        output_vec = self.opa.learn_embeddings(self.input_file)
        evaluations = self.opa.evaluate(output_vec)
		# Evaluations could be a dictionary or a sequence of metrics names
        self.assertIsInstance(evaluations, dict)
		#self.assertIn("f1", evaluations)
		#self.assertIn("MRR", evaluations)
        self.assertIn("cosine_similarity", evaluations)

		
        self.assertIsInstance(evaluations['cosine_similarity'], float)
		
		
if __name__ == '__main__':
    unittest.main()
