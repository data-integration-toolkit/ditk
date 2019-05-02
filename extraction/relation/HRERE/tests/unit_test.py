import unittest
from os import sys, path 

sys.path.append("..")
from hrere import hrere
res = hrere()
    


class TestImputationMethods(unittest.TestCase):

    def setUp(self):
        self.relation_extraction_method = res 
        self.input_file = "sample_input.txt"
        self.verificationErrors = []
        
       
    def test_read_dataset(self):
        success_or_fail = self.relation_extraction_method.read_dataset(self.input_file)
        success = 1
        self.assertEqual(success, success_or_fail)
        
            
    def test_data_preprocess(self):
        success_or_fail = self.relation_extraction_method.preprocess(self.input_file)
        success = 1
        self.assertEqual(success, success_or_fail)
        

    
    
if __name__ == '__main__':
	unittest.main()