import unittest
import similarity

class TestGraphSimilarity(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        self.graph_similarity = GraphSimilarity()
        self.input_file = 'sample_input_2.txt'
        self.model = load_model(path_to_model)
        self.test_list = {}
        with open(input_file, "r") as file:
        	for line in file:
        		split_line = line.split()
                test_list[split_line[0]] = split_line[1] 
        self.output_file = open('output_file.txt', 'w+')
        for entity1, entity2 in test_list.items(): 
            similarity = graph_similarity.similarity(entity1, entity2)
            output_file.write(entity1, entity2, similarity)
    

    def test_similarity_len(self):
        #test if similarity returns a list with three elements
        size_equals_3 = True  
        with open(output_file, "r") as file:
            for line in file:
                split_line = line.split()
                split_line = list(filter(None, split_line)) 
                if len(split_line) != 3:
                    size_equals_3 = False 

        assertTrue(size_equals_3)



    def test_similarity_val(self):
        #test if all similarity scores are between 0 and 1, assuming the first test has passed 
        similarity_0_1 = True 
        with open(output_file, "r") as file:
            for line in file:
                split_line = line.split()
                split_line = list(filter(None, split_line))
                similarity = split_line[2]
                if similarity>1 or similarity<0:
                    similarity_0_1 = False

        assertTrue(similarity_0_1)


if __name__ == '__main__':
    unittest.main()