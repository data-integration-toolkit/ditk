import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcdsat import mcdsat

class TestTextEmbeddingMethods(unittest.TestCase):

    def setUp(self):
        self.re = mcdsat() #Your implementation of TextEmbedding
        self.views_file = "sample_views.txt"
        # f = open(self.input_token1)
        self.query_file = "sample_query.txt"
        self.c2d_path = "../c2d/c2d_linux"
        self.models = "../dnnf-models/models"

    def test_generate_rewritings(self):
        self.re.read_input(viewsFile=self.views_file, queryFile=self.query_file, c2d_path=self.c2d_path, models=self.models)
        mcds = self.re.generate_MCDs()
        self.assertTrue(str(mcds))
        self.assertTrue(len(mcds) > 1)

if __name__ == '__main__':
    unittest.main()
