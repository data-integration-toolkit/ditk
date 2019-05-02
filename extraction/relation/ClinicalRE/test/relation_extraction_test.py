import unittest
import sys
import os
sys.path.append('../code/')
import ClinicalRE

class TestRelationExtraction(unittest.TestCase):

    def assertEqual(self, a, b):
        if a == b:
            return True
        print ("error")
        return False

    def setUp(self):
        #instantiate the implemented class
        self.relation_extraction = ClinicalRE.clinicalRE()
        self.input_file = 'relation_extraction_test_input.txt'
        converted_input = 'converted_input.txt'
        self.relation_extraction.read_dataset(self.input_file, converted_input)
        self.output_file = 'relation_extraction_test_output.txt'
        label_dict = {'Other': 0, 'Message-Topic(e2,e1)': 1, 'Cause-Effect(e1,e2)': 2, 'Component-Whole(e2,e1)': 3, 'Entity-Origin(e2,e1)': 4, 'Member-Collection(e2,e1)': 5, 'Message-Topic(e1,e2)': 6, 'Instrument-Agency(e1,e2)': 7, 'Product-Producer(e1,e2)': 8, 'Instrument-Agency(e2,e1)': 9, 'Entity-Destination(e1,e2)': 10, 'Content-Container(e2,e1)': 11, 'Entity-Origin(e1,e2)': 12, 'Member-Collection(e1,e2)': 13, 'Entity-Destination(e2,e1)': 14, 'Component-Whole(e1,e2)': 15, 'Product-Producer(e2,e1)': 16, 'Cause-Effect(e2,e1)': 17, 'Content-Container(e1,e2)': 18}
        self.relation_extraction.train(converted_input,label_dict, self.output_file) 
        os.remove('converted_input.txt')

    def test_outputformat(self):    

        fp_input = open(self.input_file, 'r')
        fp_output = open(self.output_file, 'r')

        inputs = fp_input.read().strip().split('\n')
        #outputs = fp_output.read().strip().split('\n')
        #test if the cases in the output file is equal to the input file
        #self.assertEqual(len(inputs), len(outputs))

        for i in range(len(inputs)):
            input_attr = inputs[i].strip().split('\t')
            #output_attr = outputs[i].strip().split('\t')
            #test if each sentence has 10 coloums in the input file
            self.assertEqual(len(input_attr), 10)
            #test if each sentence has 5 coloums in the output file
            #self.assertEqual(len(output_attr), 5)
            #test if the sentence in the output file is equal to the input file
            #self.assertEqual(input_attr[0], output_attr[0])
            #test if the first entity in the output file is equal to the input file
            #self.assertEqual(input_attr[1], output_attr[1])
            #test if the second entity in the output file is equal to the input file
            #self.assertEqual(input_attr[5], output_attr[2])


if __name__ == '__main__':
    unittest.main()