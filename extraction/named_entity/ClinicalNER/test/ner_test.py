import unittest
import sys
import os
sys.path.append('../code/')
sys.path.append('../code/code/')
import ClinicalNER

class TestNERMethods(unittest.TestCase):

    def assertEqual(self, a, b):
        if a == b:
            return True
        print ("error")
        print (a, b)
        return False

    def setUp(self):
        self.ner = ClinicalNER.clinicalNER() #Your implementation of NER
        self.input_file = 'ner_test_input.txt'
        test_txt_path = "test.txt"
        test_con_path = "test.con"
        self.ner.read_dataset(self.input_file, test_txt_path, test_con_path)
        model_path = "../code/models/CoNLL.model"
        prediction_dir = "../code/data/test_predictions/"
        prediction_path = self.ner.predict(model_path, test_txt_path, prediction_dir)
        output_path = "ner_test_output.txt"
        self.ner.output(test_txt_path, test_con_path, prediction_path, output_path)
        self.output_file = output_path
        os.remove(test_txt_path)
        os.remove(test_con_path)

    def test_outputformat(self):    
        fp_input = open(self.input_file, 'r')
        fp_output = open(self.output_file, 'r')

        inputs = fp_input.read().strip().split('\n\n')
        outputs = fp_output.read().strip().split('\n\n')
        #test if the cases in the output file is equal to input file
        self.assertEqual(len(inputs), len(outputs))

        for i in range(len(inputs)):

            input_words = inputs[i].strip().split('\n')
            output_words = outputs[i].strip().split('\n')
            #test if the number of the words in each case in the output file is equal to input file
            self.assertEqual(len(input_words), len(output_words))
            for j in range(len(input_words)):
                input_word_attr = input_words[j].strip().split(' ');
                output_word_attr = output_words[j].strip().split(' ');
                #test if each sample has 3 coloums in the output file
                self.assertEqual(len(output_word_attr), 3)
                #test if the word in the output file is equal to input file
                self.assertEqual(input_word_attr[0], output_word_attr[0])

if __name__ == '__main__':
    unittest.main()
