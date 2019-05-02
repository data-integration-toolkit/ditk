import unittest
import pandas as pd
from entity_linkage.blocking import Blocking


class TestBlockingMethods(unittest.TestCase):

	def setUp(self, input_files=['sample_input1.csv.txt', 'sample_input2.csv.txt']):
		self.blocking_method = Blocking() # initialize your Blocking method
		self.input_files = input_files

	def test_read_dataset(self):
		data_frames = self.blocking_method.read_dataset()
		self.assertIsInstance(data_frames, pd.DataFrame)

	def test_predict(self):
		predictions = self.blocking_method.predict()
		# evaluate whether predictions follow the Lists of Lists of tuples format, e.g.:
		self.assertIsInstance(predictions,list)

	def test_evaluate(self):
		precision, recall, reduction_ratio = self.blocking_method.evaluate()
		self.assertIsInstance(precision, float)
		self.assertIsInstance(recall, float)
		self.assertIsInstance(reduction_ratio, float)

if __name__ == '__main__':
    unittest.main()