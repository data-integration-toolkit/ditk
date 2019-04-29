# -*- coding: utf-8 -*-
from modules.log_config import LOG

from modules.configs import *
from gensim.models.keyedvectors import KeyedVectors
from modules.configs import *
import os
import ntpath
import numpy as np

PRE_EMBEDDING_MATRIX_DIR = 'embedding_matrix'


def load_embedding_matrix(dataset_name, word_index, embedding_dim=300):
    LOG.info('Loading embedding model from %s', EMBEDDING_FILE)
    vocab_size = len(word_index) + 1
    embedding_matrix_file = dataset_name + '_' + EMBEDDING_NAME + '_' + str(vocab_size) + '.npy'
    embedding_cache_file = os.path.join(BASE_PATH+PRE_EMBEDDING_MATRIX_DIR, embedding_matrix_file)

    # Verify if cache exists
    if os.path.exists(embedding_cache_file):
        LOG.info('Loading existing embedding matrix')
        embedding_matrix = np.loadtxt(embedding_cache_file)
    else:
        LOG.info('File %s not found. Loading new embedding matrix from: %s. Embedding binary: %s' % (embedding_matrix_file, EMBEDDING_NAME, EMBEDDING_BINARY))
        embedding_model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=EMBEDDING_BINARY)

        embedding_matrix = 1 * np.random.randn(vocab_size, embedding_dim)  # This will be the embedding matrix
        embedding_matrix[0] = 0  # So that the padding will be ignored
        unk_tokens = 0

        LOG.info('Creating the embedding matrix')
        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue
            if word in embedding_model.vocab:
                embedding_vector = embedding_model[word]
                #print("Embedding vector (%s, %s)" % (word, embedding_vector))
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
                else:
                    embedding_matrix[idx] = 0
                    unk_tokens += 1

        LOG.info('Embedding matrix as been created, removing embedding model from memory')
        LOG.info('Unknown tokens = ' + str(unk_tokens))
        del embedding_model
        LOG.info('Saving matrix in file ' + embedding_cache_file)
        np.savetxt(embedding_cache_file, embedding_matrix)

    return embedding_matrix
