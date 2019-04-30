from model.data_utils import NERDataset
from model.sal_blstm_oal_crf_model import SAL_BLSTM_OAL_CRF_Model
from model.config import Config
import os


def main():
    # create instance of config
    config = Config()
    config.filename_train = "../datasets/ritter2011/train"
    config.filename_dev = "../datasets/ritter2011/train"
    config.filename_test = "../datasets/ritter2011/train"

    config.filename_chars = config.filename_chars.replace("source", "target")
    config.filename_glove = config.filename_glove.replace("source", "target")
    config.filename_tags = config.filename_tags.replace("source", "target")
    config.filename_words = config.filename_words.replace("source", "target")

    config.dir_model = config.dir_model.replace("source", "target")
    config.dir_output = config.dir_output.replace("source", "target")
    config.path_log = config.path_log.replace("source", "target")


    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[0])
    # build model
    model = SAL_BLSTM_OAL_CRF_Model(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = NERDataset(config.filename_test, config.processing_word,
                       config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)

if __name__ == "__main__":
    main()
