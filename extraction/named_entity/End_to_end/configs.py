# -*- coding: utf-8 -*-

EMBEDDING_DIM = 100

## test-output file
OUTPUT_FILE = 'conll2003-test.txt'

## sample_test
## TEST_FILE = 'ner_test_input.txt'

## conll 20003
TRAINING_FILE = "data/train.txt"  
VALIDATION_FILE = "data/valid.txt"
MODEL_FILE = "model_weights.hdf5"
TEST_FILE = "data/test.txt"

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

GLOVE_EMBEDDINGS = "data/glove.6B.100d.txt"
BATCH_SIZE = 10
EPOCHS = 1
MAX_CHARS = 40
CHAR_EMBDS_DIM = 30
POOL_SIZE = 40
FILTER_SIZE = 3
NO_OF_FILTERS = 30
DICT_FILE = "data/dicts.txt"
MAX_SEQ_LEN = 150
EMBEDDINGS_FILE = "data/embds.npy"
DROPOUT = 0.5