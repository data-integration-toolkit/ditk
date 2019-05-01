import logging

from extraction.named_entity.UOI import Ner

training_sentence = None
training_tag = None
validate_sentence = None
validate_tag = None
uoi = Ner.UOI()


def read_dataset():
    logging.info("begin test read dataset")
    train_sen, train_tag, val_sen, val_tag = uoi.read_dataset(input_files=['/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/train.txt',
                                                                           '/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/valid.txt'],
                                                              embedding='/Users/liyiran/ditk/extraction/named_entity/UOI/embedding/glove.6B.100d.txt')
    logging.info('length of train sentences: ' + str(len(train_sen)))
    logging.info('length of train tag: ' + str(len(train_tag)))
    logging.info('length of validating sentences: ' + str(len(val_sen)))
    logging.info('length of validating tag: ' + str(len(val_tag)))
    logging.info("test read dataset")


def train():
    logging.info("begin test train")
    uoi.train(data=None)
    logging.info("test train")


def predict():
    logging.info('test predict')
    predicts = uoi.predict('/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/test.txt')
    logging.info(predicts)
    logging.info("finish predict")


def convert_ground_truth():
    # EU NNP B-NP B-ORG
    # rejects VBZ B-VP O
    # German JJ B-NP B-MISC
    # call NN I-NP O
    # to TO B-VP O
    # boycott VB I-VP O
    # British JJ B-NP B-MISC
    # lamb NN I-NP O
    # . . O O
    result = uoi.convert_ground_truth('/Users/liyiran/ditk/extraction/named_entity/UOI/CoNNL2003eng/test.txt')
    logging.info(result[0:5])
    logging.info("test convert to ground truth")


def evaluation():
    logging.info("test evaluation")
    precision, recall, f1 = uoi.evaluate()
    logging.info("precision: " + str(precision))
    logging.info("recall: " + str(recall))
    logging.info("f1: " + str(f1))
    logging.info("finish evaluation")


if __name__ == '__main__':
    read_dataset()
    train()
    predict()
    convert_ground_truth()
    evaluation()
