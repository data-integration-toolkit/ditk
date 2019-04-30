# -*- coding: utf-8 -*-

BASE_PATH = '/content/drive/My Drive/Text Similarity/semantic-neural-network/'
DATA_PATH = '/content/drive/My Drive/Text Similarity/semantic-neural-network/datasets/'
RESULTS_DIR = '/content/drive/My Drive/Text Similarity/semantic-neural-network/results/'


SICK_FILE = BASE_PATH + 'sick_2014/SICK_complete.txt'
SICK_PRETRAIN_FILE = BASE_PATH + 'sick_2014/pretrain_sts.txt'
SICK_TRAIN_FILE = BASE_PATH + 'sick_2014/SICK_train.txt'
SICK_DEV_FILE = BASE_PATH + 'sick_2014/SICK_trial.txt'
SICK_TEST_FILE = BASE_PATH + 'sick_2014/SICK_test_annotated.txt'
SICK_AUGMENTED_FILE = BASE_PATH + 'sick_2014/augmented.tsv'
SICK_AUGMENTED_NOUN_FILE = BASE_PATH + 'sick_2014/augmented_noun.tsv'

SemEval2014_TRAIN_FILE = BASE_PATH + 'SemEval2014/SemEval2014_train.txt'
SemEval2014_DEV_FILE = BASE_PATH + 'SemEval2014/SemEval2014_dev.txt'
SemEval2014_TEST_FILE = BASE_PATH + 'SemEval2014/SemEval2014_test.txt'

SemEval2017_TRAIN_FILE = BASE_PATH + 'SemEval2017/SemEval2017_train.csv'
SemEval2017_DEV_FILE = BASE_PATH + 'SemEval2017/SemEval2017_dev.csv'
SemEval2017_TEST_FILE = BASE_PATH + 'SemEval2017/SemEval2017_test.csv'


MSRP_FILE = BASE_PATH + 'msrp/msrp.tsv'

WORD2VEC_FILE_PT = BASE_PATH + 'word2vec/word2vec_pt.bin'
WORD2VEC_FILE = BASE_PATH + 'word2vec/GoogleNews-vectors-negative300.bin'
GLOVE_FILE = 'C:\dev_env\ml\datasets\glove.6B\\glove.6B.300d.gensim.txt'
FAST_TEXT_FILE = BASE_PATH + '/fasttext_english/wiki.en.vec'
FAST_TEXT_FILE_2 = BASE_PATH + '/fasttext_english/crawl-300d-2M.vec'

ASSIN_TRAIN_FILE = BASE_PATH + 'assin/assin-train.xml'
ASSIN_VAL_FILE = BASE_PATH + 'assin/assin-dev.xml'
ASSIN_TEST_FILE = BASE_PATH + 'assin/assin-test.xml'

EMBEDDING_FILE = WORD2VEC_FILE
EMBEDDING_NAME = "WORD2VEC"
EMBEDDING_BINARY = EMBEDDING_FILE == WORD2VEC_FILE or EMBEDDING_FILE == WORD2VEC_FILE_PT
EMBEDDING_DIM = 300  # dimension of the word embedding vectors"

LSTM_HIDDEN_LAYERS = 50  # by the paper

BATCH_SIZE = 32

PRETRAIN = False
PRETRAIN_EPOCHS = 66
TRAIN_EPOCHS = 300

LR = 0.5

DROPOUT = 0.0
RECURRENT_DROPOUT = 0.0

REMOVE_STOPWORDS = False

FIT_VERBOSE = 1
