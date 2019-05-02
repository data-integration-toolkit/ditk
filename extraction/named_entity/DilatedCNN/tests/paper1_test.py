import logging
import unittest

from extraction.named_entity.DilatedCNN import Ner


class Test1(unittest.TestCase):
    dilated = None

    @classmethod
    def setUpClass(cls):
        cls.dilated = Ner.DilatedCNN()
        logging.basicConfig(level=logging.INFO)
        logging.info("set up class")

    def read_dataset(self):
        logging.info('begin read dataset')

        self.dilated.read_dataset(input_files=['/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/train.txt',
                                               '/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/valid.txt',
                                               '/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/test.txt'],
                                  embedding='/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/embedding/glove.6B.100d.txt')
        logging.info('end read dataset')

    def train(self):
        logging.info('start to train')
        self.dilated.train(data=None)
        logging.info('finish to train')

    def predict(self):
        logging.info('start to predict')
        logging.info(self.dilated.predict('/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/test.txt'))
        logging.info('finish to predict')

    def convert_ground_truth(self):
        logging.info('start to test convert to ground truth')
        result = self.dilated.convert_ground_truth('/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/test.txt')
        logging.info(result[0:5])
        logging.info('finish to test convert to ground truth')

    def evaluation(self):
        logging.info('start to test evaluation method')
        logging.info(self.dilated.evaluate())
        logging.info('evaluation done')

    def test(self):
        self.read_dataset()
        self.train()
        self.predict()
        self.convert_ground_truth()
        self.evaluation()
