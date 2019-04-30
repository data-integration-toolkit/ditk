import unittest
import relation
import linecache

class TestRelationExtraction(unittest.TestCase):

    def setUp(self):
        #instantiate the implemented class
        self.relation_extraction = RelationExtraction()
        self.input_file = 'sample_input.csv.txt'
        self.output_file = relation_extraction.tokenize(input_file, 1)


    def test_tokenize(self):
        #test if the third element in the output file is equal to 'was'
        input_third_elem = 'was'
        output_third_elem = linecache.getline(output_file, 2)
        self.assertEqual(input_third_elem, output_third_elem)



if __name__ == '__main__':
    unittest.main()
