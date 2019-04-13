import unittest
import similarity

class TestGraphSimilarity(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        self.graph_similarity = GraphSimilarity()
        self.input_file = 'sample_input.txt'
    

    @unittest.expectedFailure
    def test_load_model(self):
        #test if it succesfully loads the saved model 
        self.assertRaises(graph_similarity.load_model(input_file), IOError)


if __name__ == '__main__':
    unittest.main()