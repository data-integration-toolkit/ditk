import os

# Project directory
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

############################################################
# Data Files

############################################################
# Embedding Files

_EMBEDDINGS_DIR   = _ROOT_DIR + '/embeddings'

W2V_TWITTER_EMB_GODIN = _EMBEDDINGS_DIR + '/twitter/word2vec_twitter_model.bin'
GAZET_EMB_ONE_CHECK   = _EMBEDDINGS_DIR + '/gazetteers/one.token.check.emb'

############################################################
# Global Tokens

URL_TOKEN   = '<URL>'
TAG_TOKEN   = '<TAG>'
PUNCT_TOKEN = '<PUNCT>'
EMOJI_TOKEN = '<EMOJI>'
UNK_TOKEN   = '<UNK>'
PAD_TOKEN   = '<PAD>'

#########################################################
PREDICTIONS_DIR = _ROOT_DIR + '/predictions/'
# PREDICTIONS_DIR = '/raid/data/gustavoag/ner/emnlp17/predictions/'

NN_PREDICTIONS  = PREDICTIONS_DIR + 'network.tsv'
CRF_PREDICTIONS = PREDICTIONS_DIR + 'crfsuite.tsv'

############################################################
def _test_paths():
    pass

if __name__ == '__main__':
    _test_paths()
