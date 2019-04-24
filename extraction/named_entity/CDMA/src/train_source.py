from model.data_utils import NERDataset
from model.blstm_crf_model import BLSTM_CRF_Model
from model.config import Config
import os


def main():
    # create instance of config
    config = Config()
    config.filename_train = "../datasets/ontonotes-nw/train"
    config.filename_dev = "../datasets/ontonotes-nw/dev"
    config.filename_test = "../datasets/ontonotes-nw/test"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[0])
    # build model
    model = BLSTM_CRF_Model(config)
    model.build()

    # create datasets
    train = NERDataset(config.filename_train, config.processing_word,
                       config.processing_tag, config.max_iter)

    dev   = NERDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
