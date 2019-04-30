""" This contains the function run_experiment, which can be used to run
the experiments in the COLING paper that do not involve neural networks.

To run the experiments involving the BiLSTM-CRFs, use train_bilstm_model.py
to train the model on the source dataset and save the model, and
load_pretrained.py to load the model and fine-tune on the target dataset.

"""
import os
import random

try:
    import configparser
except:
    import configparser as ConfigParser

import ditk_converter_utils

from nltk.data import path as nltk_data_path
nltk_data_location = os.getenv('NLTK_DATA_PATH')
if nltk_data_location is not None:
    nltk_data_path.append(nltk_data_location)

from train_bilstm import TrainBiLSTM
from train_crf import DomainAdaptation
import utils
from utils import _getlist

tgt_train_shuffle_seed = 'tgt_train_shuffle_seed'
src_train_shuffle_seed = 'src_train_shuffle_seed'
tgt_train_length = 'tgt_train_length'
tgt_test_length = 'tgt_test_length'
src_train_length = 'src_train_length'

class Experiment(object):

    def __init__(self):

        self.transfer_bilstm = TrainBiLSTM()
        self.D = DomainAdaptation(verbose=False)
        self.transfermethod = None

        self.ent_excluded = set()
        self.parameters = {}
        self.classifier = None

        self.dataset = 'DITK'
        self.D.dataset = self.dataset

        self.src_loaded = False
        self.src_train = None
        self.src_test = None
        self.src_entity_list = None
        self.src_embedding_files = None

        self.tgt_entity_list = None
        self.tgt_embedding_files = None
        self.tgt_train = None
        self.tgt_test = None

    def prep_data(self, data, dataset, embedding_files, entity_list, annotations_format='IOB1'):

        # Loads config for the specified experiment i.e names of file configs to be loaded from file_locations.cfg.
        DIR = 'experiments/' + self.dataset
        config_file = os.path.join(DIR, self.dataset + '.cfg')
        config = configparser.RawConfigParser(allow_no_value=True)
        config.read(config_file)

        if dataset == 'src':
            self.parameters[src_train_shuffle_seed] = config.get('corpora', 'src_train_shuffle_seed')
            if config.get('split', 'src_train_length'): self.parameters[src_train_length] = config.get('split', 'src_train_length')

            ###############
            # SOURCE DATA #
            ###############
            corpora = data['train'] + data['dev']
            corpora = self.convert_data(corpora)
            corpora = utils.read_conll_ditk(corpora,annotations_format)
            self.src_train = [sent for sent in corpora]

            corpora = data['test']
            corpora = self.convert_data(corpora)
            corpora = utils.read_conll_ditk(corpora,annotations_format)
            self.src_test = [sent for sent in corpora]

            self.src_embedding_files = embedding_files
            self.src_entity_list = ditk_converter_utils.get_ne_tags(data['test'] + data['train'] + data['dev']) if entity_list is None else entity_list

            self.src_loaded = True

        else:
            self.parameters[tgt_train_shuffle_seed] = config.get('corpora', 'tgt_train_shuffle_seed')
            if config.get('split', 'tgt_test_length'): self.parameters[tgt_test_length] = config.get('split', 'tgt_test_length')
            if config.get('split', 'tgt_train_length'): self.parameters[tgt_train_length] = config.get('split', 'tgt_train_length')
            self.classifier = config.get('algorithm', 'classifier')
            self.transfermethod = config.get('algorithm', 'transfer')
            ent_excluded = _getlist(config, 'evaluation', 'excluded')
            if ent_excluded is not None:
                self.ent_excluded = set(ent_excluded)

            ###############
            # TARGET DATA #
            ###############
            corpora = data['train'] + data['dev']
            corpora = self.convert_data(corpora)
            corpora = utils.read_conll_ditk(corpora,annotations_format)
            self.tgt_train = [sent for sent in corpora]

            corpora = data['test']
            corpora = self.convert_data(corpora)
            corpora = utils.read_conll_ditk(corpora,annotations_format)
            self.tgt_test = [sent for sent in corpora]

            self.tgt_entity_list = ditk_converter_utils.get_ne_tags(data['test'] + data['train'] + data['dev']) if entity_list is None else entity_list
            self.tgt_embedding_files = embedding_files

    def train(self, tgt_data, **kwargs):

        TRAIN_DEV_SPLIT = 0.8  # 80%/20% train/dev split.

        #####################################################################################
        # Transform the data as per TRANSFER params (seed shuffle, target train length ttl) #
        #####################################################################################
        self.resolve_user_parameters(kwargs)

        if self.src_loaded:
            # To shuffle or not to shuffle.
            if src_train_shuffle_seed in self.parameters:
                self.parameters[src_train_shuffle_seed] = self.parameters[src_train_shuffle_seed]
                self.src_train = self.shuffle_data(self.src_train, self.parameters[src_train_shuffle_seed])

            # Possibly use only first part of src training data.
            if src_train_length in self.parameters:
                self.src_train = self.src_train[:int(self.parameters[src_train_length])]

        corpora = self.convert_data(tgt_data['test'])
        corpora = utils.read_conll_ditk(corpora)
        tgt_test = [sent for sent in corpora]

        corpora = self.convert_data(tgt_data['train'])
        corpora = utils.read_conll_ditk(corpora)
        tgt_train = [sent for sent in corpora]

        if tgt_train_shuffle_seed not in self.parameters:
            self.parameters[tgt_train_shuffle_seed] = 0  # 0 means no shuffle

        # Possibly use only first part of tgt testing data.
        if tgt_test_length in self.parameters:
            tgt_test = tgt_test[:int(self.parameters[tgt_test_length])]

        if tgt_train_length not in self.parameters:
            self.parameters[tgt_train_length] = len(tgt_train)

        # Will get shuffle seed from the input or a single seed from the cfg file.
        # ttl = target training length. will be fixed to a single value from input/cfg
        # transfer method = single value from file/input from user
        tgt_trainall_shuff = self.shuffle_data(tgt_train, self.parameters[tgt_train_shuffle_seed])
        tgt_train, unused = self.split_corpus(tgt_trainall_shuff, self.parameters[tgt_train_length])

        if self.transfermethod[:4] == 'lstm':
            if not self.tgt_embedding_files:
                raise FileNotFoundError('GLove embedding file path absent. Cannot train LSTM')

            self.transfer_bilstm.set_embedding_file_path(self.tgt_embedding_files)
            if self.src_loaded:
                src_train_, src_dev_ = self.split_corpus(self.src_train, TRAIN_DEV_SPLIT)
                src_DL_data = {'train': src_train_, 'dev': src_dev_, 'test': self.src_test}
                self.transfer_bilstm.load_src_data(src_DL_data)

            tgt_train_, tgt_dev_ = self.split_corpus(tgt_train, TRAIN_DEV_SPLIT)
            tgt_DL_data = {'train':tgt_train_,'dev': tgt_dev_, 'test': tgt_test}
            self.transfer_bilstm.load_targ_data(tgt_DL_data)

            self.transfer_bilstm.load_data_embeddings()
            self.transfer_bilstm.train_src(self.max_epoches)
            self.transfer_bilstm.train_target(self.max_epoches)

        else:

            src_train = None
            if self.src_loaded:
                src_train = utils.attach_domain(self.src_train, 'src')

            tgt_train = utils.attach_domain(tgt_train, 'tgt')
            tgt_test = utils.attach_domain(tgt_test, 'tgt')
            self.D.load_data(src_train, tgt_train, tgt_test)

            if self.transfermethod[:4] == 'pred':
                if not self.src_loaded:
                    print('Source data not loaded. Cannot perform pred transfer. Switching to tgt.')
                    self.D.set_parameters('tgt', self.classifier)
                    self.D.train('tgt', self.classifier)

                params = self.method_param_mappings(self.transfermethod)
                self.D.set_parameters('pred', self.classifier)
                self.D.train('pred', self.classifier, **params)
            else:
                self.D.set_parameters(self.transfermethod, self.classifier)
                self.D.train(self.transfermethod, self.classifier)

    def predict(self, test_data):

        TRAIN_DEV_SPLIT = 0.8

        corpora = self.convert_data(test_data)
        corpora = utils.read_conll_ditk(corpora)
        test_data = [sent for sent in corpora]

        if self.transfermethod[:4] == 'lstm':
            if self.src_loaded:
                src_train_, src_dev_ = self.split_corpus(self.src_train, TRAIN_DEV_SPLIT)
                src_DL_data = {'train': src_train_, 'dev': src_dev_, 'test': self.src_test}
                self.transfer_bilstm.load_src_data(src_DL_data)

            tgt_train_, tgt_dev_ = self.split_corpus(self.tgt_train, TRAIN_DEV_SPLIT)
            tgt_DL_data = {'train':tgt_train_,'dev': tgt_dev_, 'test': test_data}
            self.transfer_bilstm.load_targ_data(tgt_DL_data)
            predicted = self.transfer_bilstm.predict(test_data)
        else:
            predicted = self.D.predict(test_data)
        formatted_pred = self.format_predict(predicted)
        return formatted_pred

    def eval(self, predicted, actual):
        write_output('ner_test_output.txt',actual,predicted)
        predicted = self.format_for_eval(predicted)
        actual = self.format_for_eval(actual)

        if self.transfermethod[:4] == 'lstm':
            score = self.transfer_bilstm.eval(actual,predicted)
        else:
            score = self.D.eval(predicted,actual)

        macroP, macroR, macroF1 = score.macroPRF1()
        return macroP*100, macroR*100, macroF1*100

    def save_model(self, location='models/current'):
        exp_params = {}
        exp_params['transfer_method'] = self.transfermethod
        if self.transfermethod[:4] == 'lstm':
            self.transfer_bilstm.save_tgt_model(location)
        else:
            self.D.save_model_external(location)
        utils.save_params(exp_params,'experiment',location)

    def load_model(self, location='models/current'):
        exp_params = utils.load_params('experiment', location)
        self.transfermethod = exp_params['transfer_method']
        if self.transfermethod[:4] == 'lstm':
            self.transfer_bilstm.load_tgt_model(location)
        else:
            self.D.load_model_external(location)

############################
#### Other util functions ##
############################
    def shuffle_data(self, corpus, seed):
        """ Shuffle the data. corpus is list of lists (sentences) to be shuffled.

        Note: I will use the convention that seed 0 means no shuffle. If seed is
        None that also means no shuffle will occur.

        """
        if seed not in {None, 0}:
            random.seed(seed)
            shuffled = random.sample(corpus, len(corpus))
        else:
            shuffled = corpus

        return shuffled

    def split_corpus(self, data, train_amount, test_length=None):
        """ Split the sentences into train and test sets. If test_length
        is None, then the split is:

        train = the first 'train_length' number of sentences.
        test = the rest.

        If test_length is given, the split is:

        train = the first 'train_length' number of sentences.
        test = the last 'test_length' number of sentences.

        NOTE: train_amount + test_length must be less than the total number of
        sentences.

        """
        if 0 <= train_amount <= 1:
            train_length = int(len(data) * train_amount)
        else:
            train_length = train_amount  # number of sentences in training set
        train = data[:train_length]

        if test_length is None:
            test = data[train_length :]
        else: # test_length is number of testing sentences
            if train_length + test_length > len(data):
                raise ValueError("The corpus is not long enough to have that much training & testing data.")
            test = data[len(data)-test_length : ]
        return train, test

    def method_param_mappings(self, method):
        if method == 'pred':
            params = {'with_cca': False, 'no_prefix':False}

        if method == 'pred-no_prefix':
            params = {'with_cca': False, 'no_prefix':True}

        if method == 'predCCA':
            params = {'with_cca': True, 'no_prefix':False, 'exclude_O':False}

        if method == 'predCCA-no_prefix':
            params = {'with_cca': True, 'no_prefix':True, 'exclude_O':False}

        if method == 'predCCA-no_prefix-excludeO':
            params = {'with_cca': True, 'no_prefix':True, 'exclude_O':True}

        if method == 'predCCA-excludeO':
            params = {'with_cca': True, 'no_prefix':False, 'exclude_O':True}

        return params

    def convert_data(self, data):
        data = ditk_converter_utils.convert_to_line_format(data)
        data = ditk_converter_utils.ditk_ner_to_conll2003(data)
        data = ditk_converter_utils.convert_to_line_format(data)
        return data

    def resolve_user_parameters(self, kwargs):
        if 'classifier' in kwargs: self.classifier = kwargs['classifier']
        if 'transfer_method' in kwargs: self.transfermethod = kwargs['transfer_method']
        if self.transfermethod not in ['lstm', 'predCCA', 'pred', 'tgt']: raise ValueError('Invalid transfer method!',['lstm', 'predCCA', 'pred', 'tgt'])
        if self.classifier not in ['CRF', 'naivebayes', 'IIS', 'GIS']: raise ValueError('Invalid classifier! Valid values:', ['CRF', 'naivebayes', 'IIS', 'GIS'])
        if 'max_epoches' in kwargs: self.max_epoches = int(kwargs['max_epoches'])
        if 'shuffle_seed' in kwargs: self.parameters[tgt_train_shuffle_seed] = int(kwargs['shuffle_seed'])
        if 'excluded_tags' in kwargs: self.ent_excluded = set(kwargs['excluded_tags'])
        if 'src_data' in kwargs:
            data = kwargs['src_data']
            corpora = data['train'] + data['dev']
            corpora = self.convert_data(corpora)
            corpora = utils.read_conll_ditk(corpora)
            self.src_train = [sent for sent in corpora]

            corpora = data['test']
            corpora = self.convert_data(corpora)
            corpora = utils.read_conll_ditk(corpora)
            self.src_test = [sent for sent in corpora]
            self.src_loaded = True

    def format_predict(self, predicted):
        formatted_predict = []
        for sent in predicted:
            for token_data in sent:
                formatted_predict.append((None, None, token_data[0][0], token_data[1]))
            formatted_predict.append(())
        return formatted_predict

    def format_for_eval(self, data):
        converted_data = []
        sent = []
        for token in data:
            if len(token) != 0:
                zz = tuple([token[2]])
                sent.append((zz, token[3]))
            else:
                converted_data.append(sent)
                sent = []

        return converted_data

def write_output(file, actual, predicted):
    with open(file, 'w') as f:
        for idx, sent in enumerate(predicted):
            if len(sent) == 0:
                if idx+1 != len(predicted):
                    f.write('\n')
            else:
                act_tag = actual[idx][3]
                pred_tag = sent[3]
                word = sent[2]
                f.write("%s %s %s\n" % (word, act_tag, pred_tag))
