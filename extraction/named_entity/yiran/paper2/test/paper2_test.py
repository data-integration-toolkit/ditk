import unittest
import logging

from extraction.named_entity.yiran.paper2 import Ner

# import unittest
# import pandas as pd
# from named_entity import ner 
# 
# class TestNERMethods(unittest.TestCase):
# 
#     def setUp(self):
#         self.ner = Ner() #Your implementation of NER
#         self.input_file = 'path_to_sample_input.txt'
#         self.output_file = ner.main(input_file)
# 
#     def row_col_count(file_name):
#         df = pd.read_csv(file_name,delim=' ')
#         return df.shape
# 
#     def test_outputformat(self):
#         input_row_count = row_col_count(input_file)[0]
#         input_col_count = row_col_count(input_file)[1]
#         output_row_count = row_col_count(output_file)[0]
#         output_col_count = row_col_count(output_file)[1]
# 
#         self.assertEqual(input_row_count, output_row_count)
#         self.assertEqual(output_col_count, 3)
# 
# if __name__ == '__main__':
#     unittest.main()


class Test2(unittest.TestCase):
    uoi = None
    training_sentence = None
    training_tag = None
    validate_sentence = None
    validate_tag = None

    @classmethod
    def setUpClass(cls):
        cls.uoi = Ner.UOI()
        logging.basicConfig(level=logging.INFO)
        print("set up class")

    def read_dataset(self):
        print("begin test read dataset")
        train_sen, train_tag, val_sen, val_tag = self.uoi.read_dataset(inputFiles= ['/Users/liyiran/ditk/extraction/named_entity/yiran/CoNNL2003eng/train.txt', 
                                                                                    '/Users/liyiran/ditk/extraction/named_entity/yiran/CoNNL2003eng/valid.txt'])
        logging.info('length of train sentences: ' + str(len(train_sen)))
        logging.info('length of train tag: ' + str(len(train_tag)))
        logging.info('length of validating sentences: ' + str(len(val_sen)))
        logging.info('length of validating tag: ' + str(len(val_tag)))
        print("test read dataset")

    def train(self):
        print("begin test train")
        self.uoi.train(data=None)
        print("test train")

    def predict(self):
        predicts = self.uoi.predict('/Users/liyiran/csci548sp19projectner_my/paper2/CoNNL2003eng/test.txt')
        print(predicts)
        print("test predict")

    def convert_ground_truth(self):
        self.uoi.convert_ground_truth(data=['Nadim'])
        print("test convert to ground truth")

    def evaluation(self):
        self.uoi.evaluate("")
        print("test evaluation")

    def test(self):
        self.read_dataset()
        # self.train()
        # self.predict()
        # self.convert_ground_truth()
        # self.evaluation()
