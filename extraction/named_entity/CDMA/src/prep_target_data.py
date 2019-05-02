from model.config import Config
from model.data_utils import NERDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


datasets = {"c":"conll2003", "o":"ontonotes-nw", "r":"ritter2011", "w":"wnut2016"}


def main():
    # get config and processing of words
    config = Config(load=False)
    # should be source_x.txt

    # or ontonotes-nw if you like

    config.filename_train = "../datasets/ritter2011/train"
    config.filename_dev = "../datasets/ritter2011/dev"
    config.filename_test = "../datasets/ritter2011/test"

    config.filename_chars =  config.filename_chars.replace("source", "target")
    config.filename_glove = config.filename_glove.replace("source", "target")
    config.filename_tags = config.filename_tags.replace("source", "target")
    config.filename_words = config.filename_words.replace("source", "target")

    config.dir_model = config.dir_model.replace("source", "target")
    config.dir_output = config.dir_output.replace("source", "target")
    config.path_log = config.path_log.replace("source", "target")


    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = NERDataset(config.filename_dev, processing_word)
    test  = NERDataset(config.filename_test, processing_word)
    train = NERDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)
    vocab_tags.add(UNK)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim Word Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = NERDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
