import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cnn_model import CNNModel

class RelationExtractionTest(unittest.TestCase):
    def setUp(self):
        self.model = CNNModel()
        dir_path = os.path.dirname(__file__)
        dir_path = dir_path.replace(os.sep, '/')
        self.input_dir = dir_path+'/data'
        self.model_dir = dir_path+'/model'

    def test_preprocess(self):
        # remove previously generated files if exist
        for (dirpath, _, filenames) in os.walk(self.input_dir):
            for filename in filenames:
                if filename != "relation_extraction_input_train.txt" and filename != "relation_extraction_input_test.txt":
                    os.remove(dirpath+"/"+filename)
            
            break
        
        self.model.data_preprocess(self.input_dir)

        # assert files needed for the model are generated
        assert os.path.exists(self.input_dir+'/train.cln')
        assert os.path.exists(self.input_dir+'/test.cln')
        assert os.path.exists(self.input_dir+'/relations.txt')
        assert os.path.exists(self.input_dir+'/senna_words.lst')
        assert os.path.exists(self.input_dir+'/embed50.senna.npy')
    
    def test_read_dataset(self):
        self.model.data_preprocess(self.input_dir)
        self.model.read_dataset(self.input_dir)

        # assert the dict for relation types is loaded
        self.assertTrue(len(self.model.relation_types_dict)>0)
    
    def test_train(self):
        self.model.data_preprocess(self.input_dir)
        self.model.read_dataset(self.input_dir)

        self.model.train(self.input_dir, self.model_dir)

        # assert trained model exists
        assert os.path.exists(self.model_dir+'/cnn-200-50')
        assert os.path.isdir(self.model_dir+'/cnn-200-50')

if __name__ == '__main__':
    unittest.main()