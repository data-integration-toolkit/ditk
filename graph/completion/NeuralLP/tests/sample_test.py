import unittest
import pandas as pd

import sys, os
sys.path.insert(0,"/Users/tushyagautam/Documents/USC/Information_Integration/Project/Neural-LP-master/NeuralLP")

from kgc_refactor import KGC_NeuralLP


class TestGraphCompletionMethods(unittest.TestCase):

	def setUp(self, input_file):
		self.graph_completion = KGC_NeuralLP() # initialize your specific Graph Completion class
		self.input_file = input_file
        self.output_file = graph_completion.main(input_file)

	def test_read_dataset(self):
		data_ans = self.graph_completion.read_dataset()
		self.assertTrue(data_ans != None)
		# You need to make sure that the output format of
		# the read_dataset() function for any given input remains the same
		#self.assertTrue(train, list) # assert non-empty list
		#self.assertTrue(test, list) # assert non-empty list
		#self.assertTrue(dev, list) # assert non-empty list

	def test_predict(self):
		predictions = self.graph_completion.get_truths()
		# evaluate whether predictions follow a common format such as:
		# each tuple in the output likely will follow format: (subject_entity, relation, object_entity)
		self.assertTrue(predictions, list)  # assert non-empty list

	def test_evaluate(self):
		evaluations = self.graph_completion.evaluate()
		# Make sure that the returned metrics are inside a dictionary and the required keys exist
		self.assertIsInstance(evaluations, dict)
		self.assertIn("Hits@10", evaluations)
		self.assertIn("MRR", evaluations)

	def test_output_facts():
		#provide path to input file
		data = pd.read_csv(input_file,delim=' ')
		input_entity1 = data.Entity1.unique().tolist()
		input_entity2 = data.Entity1.unique().tolist()
		input_relations = data.Relation.unique().tolist()
		input_entities = input_entity1 + input_entity2
		#provide path to output file
		data = pd.read_csv(output_file,delim=' ')
		output_entity1 = data.Entity1.unique().tolist()
		output_entity2 = data.Entity1.unique().tolist()
		output_entities = output_entity1 + output_entity2
		output_relations = data.Relation.unique().tolist()
		self.assertEqual(set(input_entities),set(output_entities))
		self.assertEqual(set(input_relations),set(output_relations))

if __name__ == '__main__':
    unittest.main()