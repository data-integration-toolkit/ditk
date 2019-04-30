# -*- coding: utf-8 -*-

EMBEDDING_DIM = 100

## test-output file
OUTPUT_FILE = 'ner_test_output.txt'

## sample_test
TEST_FILE = 'ner_test_input.txt'

## conll 20003
TRAINING_FILE = "DATA/train.txt"  
VALIDATION_FILE = "DATA/valid.txt"
MODEL_FILE = "model_weights.hdf5"
#TEST_FILE = "ner_test_input.txt"

## chemdner
#TRAINING_FILE = "data/che_train.txt"  
#VALIDATION_FILE = "data/che_valid.txt"
#MODEL_FILE = "model_weights_che.hdf5"
#TEST_FILE = "data/che_test.txt"

## ontonotes
#TRAINING_FILE = "data/data.modified_train_ontonotes.txt"
#VALIDATION_FILE = "data/ontonotes_valid.txt"
#MODEL_FILE = "model_weights_che.hdf5"
#TEST_FILE = "data/data.modified_test_ontonotes.txt"

GLOVE_EMBEDDINGS = "DATA/glove.6B.100d.txt"
BATCH_SIZE = 10
EPOCHS = 1
MAX_CHARS = 40
CHAR_EMBDS_DIM = 30
POOL_SIZE = 40
FILTER_SIZE = 3
NO_OF_FILTERS = 30
DICT_FILE = "DATA/dicts.txt"
MAX_SEQ_LEN = 150
EMBEDDINGS_FILE = "DATA/embds.npy"
DROPOUT = 0.5