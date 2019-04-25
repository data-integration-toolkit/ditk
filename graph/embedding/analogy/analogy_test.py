from graph.embedding.analogy.dataset import Vocab, TripletDataset
from graph.embedding.analogy.analogy import ANALOGY
import unittest


class TestAnalogyMethods(unittest.TestCase):

    def setUp(self):
    # def setUp(self):
        self.graph_embedding = ANALOGY()  # initialize your Blocking method
        # self.input_file = input_file
        pass

    def test_read_dataset(self):
        t = 1
        self.assertTrue(t == 2)


if __name__ == '__main__':
    unittest.main()
