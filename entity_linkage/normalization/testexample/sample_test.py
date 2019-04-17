import unittest
import normalization

class TestGraphSimilarity(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        self.entity_normalization = EntityNormalization()
        self.input_file = 'sample_input.txt'
        self.model = load_model(path_to_model)
        self.test_list = []
        with open(input_file, "r") as file:
        	for line in file:
        		lines.append(line)
    

    def test_predict(self):
        #test if predicts returns at least list with two elements 
        results = predict(model, test_list)
        assertTrue(len(results)>=2)




if __name__ == '__main__':
    unittest.main()