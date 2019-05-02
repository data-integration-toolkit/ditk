from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()
    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    max_sequence_length = max(max([len(seq[0]) for seq in train]), max([len(seq[0]) for seq in dev]),
                              max([len(seq[0]) for seq in test]))

    max_word_length = max(max([len(word[0]) for seq in train for word in seq[0]]),
                          max([len(word[0]) for seq in test for word in seq[0]]),
                          max([len(word[0]) for seq in dev for word in seq[0]]))
    print(max_word_length, max_sequence_length)
    model = NERModel(config, max_word_length,max_sequence_length)
    model.build()
    model.train(train, dev)

if __name__ == "__main__":
    main()
