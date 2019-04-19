from graph.embedding.sample_test import TestGraphEmbeddingMethods
from graph.embedding.analogy.analogy import ANALOGY


class TestAnalogyMethods(TestGraphEmbeddingMethods):

    def setUp(self, input_file):
        self.graph_embedding = ANALOGY()  # initialize your Blocking method
        self.input_file = input_file

