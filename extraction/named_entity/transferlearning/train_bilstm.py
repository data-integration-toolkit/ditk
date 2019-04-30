""" This contains functions to train and test a BiLSTM-CRF on the source corpus,
and to save both the trimmed word embeddings used and the model weights,
which can be used to initialize another neural net for fine tuning.

Usage
-----

>>> max_len, we, w2i, words = get_embeddings()
>>> # Train the biLSTM-CRF, and save to disk:
>>> history, score = fit_and_test_model(max_len, we, w2i, words)

"""

#################################
# For reproducibility (though this is not entirely reproducible)
import numpy as np
import random
import tensorflow as tf

TGT_TRAINED_MODEL = 'trained_model.h5'

tf.set_random_seed(1234)
np.random.seed(42)
random.seed(12345)

#################################

import gzip
import os
from keras import optimizers
import utils

try:
    import pickle as pkl
except:
    import _pickle as pkl  # Python 3

try:
    import configparser
except:
    import configparser as ConfigParser

import embedding_utils
import evaluation
from utils import _getlist
import classifiers.lstmcrf as lc
from stratified_split import writefile

CONLLTAGSET = {'PER','LOC','ORG','MISC'}



class TrainBiLSTM():

    def __init__(self):
        self.WVDIM = '100'
        self.embeddingsPath = 'word_embeddings/glove.6B.'+self.WVDIM+'d.txt'
        self.epochs = 1
        self.tlayers = 'EL'
        self.max_len = 253
        self.MODEL_LOCATION = 'models_bilstmcrf/'
        self.MODEL_FILE  = self.MODEL_LOCATION +'final_model_100_withcase3.h5'
        self.EMBEDDINGS_FILE = self.MODEL_LOCATION + 'embeddings_1.pkl.gz'
        self.ent_excluded = set()
        self.dataset = "DITK"
        self.src_data = None
        self.tgt_data = None
        self.ttl = None
        self.seed = None

    def set_embedding_file_path(self, glove_path):
        self.embeddingsPath = glove_path

    def load_src_data(self, src_data):
        self.src_data = src_data

    def load_targ_data(self, tgt_data):
        self.tgt_data = tgt_data

    def load_data_embeddings(self):
        print("Getting vocab from various datasets...")
        src_full_data = self.src_data['train'] + self.src_data['test'] + self.src_data['dev'] if self.src_data else []
        tgt_full_data = self.tgt_data['train'] + self.tgt_data['test'] + self.tgt_data['dev']

        aggregation = src_full_data + tgt_full_data
        self.words = lc.get_word_dict(aggregation)
        self.max_len = lc.get_maxlen(aggregation)

        print("Getting word embeddings...")
        self.we, self.w2i = embedding_utils.get_word_embeddings(self.embeddingsPath, self.words)

        print("Obtaining source train data...")
        self.tags_src, self.tag2idx_src = lc.get_tag2idx(self.src_data['train'] + self.src_data['test'] + self.src_data['dev'])
        self.src_X_tr, self.src_y_tr, self.src_Xtr_ca = lc.prepare_inputs_outputs(self.src_data['train'], self.w2i,self.tag2idx_src, self.max_len)
        self.src_X_dev, self.src_y_dev, self.src_Xdev_ca = lc.prepare_inputs_outputs(self.src_data['dev'], self.w2i,self.tag2idx_src, self.max_len)

        print("Obtaining tgt train data...")
        self.tags_tgt, self.tag2idx_tgt = lc.get_tag2idx(self.tgt_data['train'] + self.tgt_data['test'] + self.tgt_data['dev'])
        self.tgt_X_tr, self.tgt_y_tr, self.tgt_Xtr_ca = lc.prepare_inputs_outputs(self.tgt_data['train'], self.w2i,self.tag2idx_tgt, self.max_len)
        self.tgt_X_dev, self.tgt_y_dev, self.tgt_Xdev_ca = lc.prepare_inputs_outputs(self.tgt_data['dev'], self.w2i,self.tag2idx_tgt, self.max_len)


        print('Saving the word embeddings for use later.')
        embeddings = {'we': self.we,
                      'w2i': self.w2i,
                      'l2i': self.tag2idx_src}

        embedding_utils.pkl_save(self.EMBEDDINGS_FILE,
                                 [embeddings],
                                 "Embeddings")

        with gzip.open(self.EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pkl.load(f)
            self.t2i = embeddings['l2i']
            self.w2i = embeddings['w2i']
            self.we = embeddings['we']

            tags = [''] * len(self.t2i)
            for k, v in list(self.t2i.items()):
                tags[v] = k


        config_file = 'experiments/' + self.dataset + '/' + self.dataset + '.cfg'
        config = configparser.RawConfigParser(allow_no_value=True)
        config.read(config_file)

        self.ent_excluded = _getlist(config, 'evaluation', 'excluded')
        if self.ent_excluded is not None:
            self.ent_excluded = set(self.ent_excluded)

        self.ttl = config.get('split', 'tgt_train_length')
        self.seed = 5

    def train_src(self, max_epoches=100):
        """ Fit and test a BiLSTM-CRF on the CONLL 2003 corpus. Return both the
        training history and the score (evaluated on the source testing file).

        """
        # NOTE Using the custom train/dev/test split.

        self.epochs = max_epoches
        model, crf = lc.make_biLSTM_casing_CRF2(len(self.tag2idx_src),
                                                len(self.w2i),
                                                self.max_len,
                                                self.we,
                                                WORDVEC_DIM=int(self.WVDIM))

        optimizer = optimizers.SGD(lr=0.005, clipvalue=5.0)

        model.compile(optimizer = optimizer,
                      loss = crf.loss_function,
                      metrics = [crf.accuracy])

        history = lc.fit_model(model, self.src_X_tr, self.src_Xtr_ca, self.src_y_tr,
                               self.src_X_dev, self.src_y_dev, self.src_Xdev_ca,
                                epochs=self.epochs, savename=self.MODEL_FILE)

    def load_pretrained_model(self, tlayers='EL'):
        """ Load a pre-trained model and decide which layers should be transfered.

        For tlayers use 'N': none, train from scratch
                        'E': transfer embedding layer only
                        'EL': transfer both embedding layer and biLSTM layer

        """

        self.model, self.crf = lc.make_biLSTM_casing_CRF2(len(self.tag2idx_tgt), len(self.w2i), self.max_len, self.we, WORDVEC_DIM=int(self.WVDIM))

        # This changes the name of the last layer so it doesn't match 'crf_1'.
        self.model.layers[-1].name = 'do_not_load_me'
        self.model.layers[-2].name = 'do_not_load_me'
        self.optimizer = optimizers.RMSprop(clipvalue=5.0)

        self.model.compile(optimizer = self.optimizer, loss = self.crf.loss_function, metrics = [self.crf.accuracy])

        if tlayers in {'E', 'EL'}:
            if tlayers == 'E':
                self.model.get_layer(name='biLSTM').name = 'do_not_load_me'
                print("Only transfer the first (word embedding) layer.")
            if tlayers == 'EL':
                print("Transfer both the word embedding and the biLSTM layer.")
            self.model.load_weights(self.MODEL_FILE, by_name=True)
        elif tlayers == 'N':
            print("Not reusing any layers. Training from scratch.")
        else:
            raise ValueError("!!")

    def train_target(self, max_epoches=100):
        self.epochs = max_epoches
        self.load_pretrained_model()
        self.history = lc.fit_model(self.model, self.tgt_X_tr, self.tgt_Xtr_ca, self.tgt_y_tr, self.tgt_X_dev, self.tgt_y_dev, self.tgt_Xdev_ca, self.epochs)
        directory = self.MODEL_LOCATION
        self.model.save(filepath=directory+TGT_TRAINED_MODEL,overwrite=True,include_optimizer=True)

    def load_saved_model(self):
        self.model, self.crf = lc.make_biLSTM_casing_CRF2(len(self.tag2idx_tgt), len(self.w2i), self.max_len, self.we, WORDVEC_DIM=int(self.WVDIM))
        self.optimizer = optimizers.RMSprop(clipvalue=5.0)
        self.model.compile(optimizer = self.optimizer, loss = self.crf.loss_function, metrics = [self.crf.accuracy])
        lc.fit_model(self.model, self.tgt_X_tr, self.tgt_Xtr_ca, self.tgt_y_tr, self.tgt_X_dev, self.tgt_y_dev, self.tgt_Xdev_ca, 0)
        self.model.load_weights(self.MODEL_LOCATION+TGT_TRAINED_MODEL, by_name=True)

    def predict(self, tgt_test):
        self.load_saved_model()
        pred = lc.predict(self.model, tgt_test, self.tag2idx_tgt, self.w2i, self.tags_tgt, self.max_len)
        return pred

    def eval(self, actual, pred):
        score = evaluation.Evaluator(pred, actual, CONLLTAGSET)
        directory = self.MODEL_LOCATION
        writefile(score.predicted, directory, 'predicted.conll', sep=' ')
        score.write_report(os.path.join(directory, 'results.txt'), self.ent_excluded)
        return score

    def load_tgt_model(self, location='models/current'):
        utils.clear_model_folder(self.MODEL_LOCATION)
        utils.update_recent_model(location,self.MODEL_LOCATION)
        self.load_saved_model()


    def save_tgt_model(self,location='models/current'):
        utils.update_recent_model(self.MODEL_LOCATION, location)
