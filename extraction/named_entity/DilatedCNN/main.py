import logging

from extraction.named_entity.DilatedCNN import Ner

dilated = Ner.DilatedCNN()


def read_dataset():
    logging.info('begin read dataset')

    dilated.read_dataset(input_files=['/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/train.txt',
                                      '/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/valid.txt',
                                      '/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/test.txt'],
                         embedding='/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/embedding/glove.6B.100d.txt')
    logging.info('end read dataset')


def train():
    logging.info('start to train')
    dilated.train(data=None)
    logging.info('finish to train')


def predict():
    logging.info('start to predict')
    logging.info(dilated.predict('/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/test.txt'))
    logging.info('finish to predict')


def convert_ground_truth():
    logging.info('start to test convert to ground truth')
    result = dilated.convert_ground_truth('/Users/liyiran/ditk/extraction/named_entity/DilatedCNN/CoNNL2003eng/test.txt')
    logging.info(result[0:5])
    logging.info('finish to test convert to ground truth')


def evaluation():
    logging.info('start to test evaluation method')
    logging.info(dilated.evaluate())
    logging.info('evaluation done')


if __name__ == "__main__":
    # execute only if run as a script
    read_dataset()
    train()
    predict()
    convert_ground_truth()
    evaluation()
