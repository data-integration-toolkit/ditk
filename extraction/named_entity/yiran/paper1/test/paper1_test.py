import logging
import unittest

from extraction.named_entity.yiran.paper1 import Ner


class Test1(unittest.TestCase):
    dilated = None

    @classmethod
    def setUpClass(cls):
        cls.dilated = Ner.DilatedCNN()

    def read_dataset(self):
        logging.info('begin read dataset')

        self.dilated.read_dataset(input_files=['/Users/liyiran/ditk/extraction/named_entity/yiran/CoNNL2003eng/train.txt',
                                               '/Users/liyiran/ditk/extraction/named_entity/yiran/CoNNL2003eng/valid.txt',
                                               '/Users/liyiran/ditk/extraction/named_entity/yiran/CoNNL2003eng/test.txt'],
                                  embedding='/Users/liyiran/ditk/extraction/named_entity/yiran/embedding/glove.6B.100d.txt')
        logging.info('end read dataset')

    def train(self):
        self.dilated.train(data=None)

    def predict(self):
        pass

    def convert_ground_truth(self):
        pass

    def evaluation(self):
        pass

    def test(self):
        self.read_dataset()
        self.train()
        self.predict()
        # self.convert_ground_truth()
        # self.evaluation()
