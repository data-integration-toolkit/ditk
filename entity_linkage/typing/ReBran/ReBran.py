#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 2019

@author: disarideb
"""

from entity_typing import entity_typing

from src.processing.utils.process_CDR_data import *
from src.processing.utils.filter_hypernyms import *
from src.processing.labled_tsv_to_tfrecords import *
from src.processing.ner_to_tfrecords import *
from src.train_multiclass_classifier import *
from src.predict import *
from src.evaluate import *


class ReBran(entity_typing):

    '''
    The constructor can be used if needed
    def __init__(self, constructor_param1):
        self.constructor_param1 = constructor_param1
    '''

    '''
    All functions declared in the abstract class to be written here too, even if you're not using it
    '''

    def read_dataset(self, file_names, options={}):

        data=[]
        lines=[line.strip() for line in open(file_names)]
        for line in lines:
            return line

        '''

        input - CDR_DevelopmentSet.PubTator.txt, CDR_TestSet.PubTator.txt, CDR_TrainingSet.PubTator.txt

        internal inputs - 2017MeshTree.txt, word_piece_vocabs, CDR_pubmed_ids

        steps:
        - use byte-pair encoding (BPE) tokenization (process_CDR_data.py)
        - split in positive and negative (filter_hypernyms.py)
        - convert processed data to tensorflow protobufs (labled_tsv_to_tfrecords.py)
        - convert ner data to tf protos (ner_to_tfrecords.py)

        output - list containing the data sets for training (ner_CDR_dev.txt,ner_CDR_train.txt, Cner_CDR_test.txt)
        '''
        #pass
        #return processed data

    def train(self, train_data, options={}) :


        '''


        def train_model(model, pos_dist_supervision_batcher, neg_dist_supervision_batcher,
                positive_train_batcher, negative_train_batcher, ner_batcher, sv, sess, saver, train_op, ner_train_op,
                string_int_maps, positive_test_batcher, negative_test_batcher, ner_test_batcher,
                positive_test_test_batcher, negative_test_test_batcher, tac_eval, fb15k_eval,
                text_prob, text_weight,
                log_every, eval_every, neg_samples, save_path,
                kb_pretrain=0, max_decrease_epochs=5, max_steps=-1, assign_shadow_ops=tf.no_op()):

        input - ner_CDR_dev.txt,ner_CDR_train.txt,

        steps - train_multiclass_classifier.py (FLAGS.mode == 'train')

        output - model.tf (optional) / saved locally


        '''
        save_path = '%s/%s' % (FLAGS.logdir, FLAGS.save_model) if FLAGS.save_model != '' else None
        train_model(train_data,
                            FLAGS.text_prob, FLAGS.text_weight,
                            FLAGS.log_every, FLAGS.eval_every, FLAGS.neg_samples,
                            save_path, FLAGS.kb_pretrain, FLAGS.max_decrease_epochs,
                            FLAGS.max_steps)

        #return model

    def predict(self, test_data, model_details=None, options={}) :
        '''
        (optional for my program structure) - may change....not able to locate any predicted file while executing the program as of now.
        input -

        output -

        '''
        print('Exporting predictions')
        export_scores(model_details, FLAGS, test_data, FLAGS.export_file)

        #sem_eval_data_predict = data_converter.Common_to_SemEval2010(test_data,'predict')

        #return predicted value/text


    def evaluate(self, test_data, prediction_data=None,  options={}) :
        '''

        input - ner_CDR_test.txt , model.tf (loaded locally)

        steps - train_multiclass_classifier.py (FLAGS.mode == 'evaluate')

        output - Precision, Recall, F-1 (overall and as per segments of Disease and Chemical)

        '''

        model=options
        print('Evaluating')
        results, _, threshold_map = relation_eval(model, FLAGS, test_data)
        print (results)

        #return evaluation tuple


    '''
    Extra individual functions

    def own_function(self):
        return 1
    '''

'''    
User Input
my_obj = ReBran()

print(my_obj.read_dataset())

print(my_obj.train())

print(my_obj.evaluate())

        
'''
