import logging
import unittest

from extraction.named_entity.UOI import Ner


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
        logging.info("set up class")

    def read_dataset(self):
        logging.info("begin test read dataset")
        train_sen, train_tag, val_sen, val_tag = self.uoi.read_dataset(input_files=['/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/train.txt',
                                                                                    '/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/valid.txt'],
                                                                       embedding='/Users/liyiran/ditk/extraction/named_entity/UOI/embedding/glove.6B.100d.txt')
        logging.info('length of train sentences: ' + str(len(train_sen)))
        logging.info('length of train tag: ' + str(len(train_tag)))
        logging.info('length of validating sentences: ' + str(len(val_sen)))
        logging.info('length of validating tag: ' + str(len(val_tag)))
        logging.info("test read dataset")

    def train(self):
        logging.info("begin test train")
        self.uoi.train(data=None)
        logging.info("test train")

    def predict(self):
        logging.info('test predict')
        predicts = self.uoi.predict('/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/test.txt')
        logging.info(predicts)
        logging.info("finish predict")

    def convert_ground_truth(self):
        # EU NNP B-NP B-ORG
        # rejects VBZ B-VP O
        # German JJ B-NP B-MISC
        # call NN I-NP O
        # to TO B-VP O
        # boycott VB I-VP O
        # British JJ B-NP B-MISC
        # lamb NN I-NP O
        # . . O O
        result = self.uoi.convert_ground_truth('/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/test.txt')
        logging.info(result[0:5])
        logging.info("test convert to ground truth")

    def evaluation(self):
        logging.info("test evaluation")
        precision, recall, f1 = self.uoi.evaluate()
        logging.info("precision: " + str(precision))
        logging.info("recall: " + str(recall))
        logging.info("f1: " + str(f1))
        logging.info("finish evaluation")

    def test(self):
        self.read_dataset()
        self.train()
        self.predict()
        self.convert_ground_truth()
        self.evaluation()
