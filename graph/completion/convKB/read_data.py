import numpy as np
from builddata import *

def read_data(data_name, split_ratio, embedding_dim):
    # Load data
    print("Loading data...")

    train, valid, test, words_indexes, indexes_words, \
    headTailSelector, entity2id, id2entity, relation2id, id2relation = build_data(filename=data_name, split_ratio=split_ratio)

    lstEmbed = []

    assert len(words_indexes) % (len(entity2id) + len(relation2id)) == 0

    print("Loading data... finished!")

    return train, valid, test, words_indexes, indexes_words, headTailSelector, entity2id, id2entity, relation2id, id2relation, lstEmbed
