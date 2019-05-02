#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:28:46 2019

@author: aravindjyothi
"""

import initial
import cnn_rnn_model
import plot_pr
import tester
from os import sys, path 

relation_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
AtNRE_dir = path.dirname(path.dirname(path.abspath(__file__)))
src_dir = path.dirname(path.abspath(__file__))

sys.path.append(relation_dir)
sys.path.append(src_dir)
sys.path.append(AtNRE_dir)

from relation_extraction import RelationExtraction

class AtNRE(RelationExtraction):

    def __init__(self):
        pass

    def read_dataset(self, input_file):
        k = initial.init_entityebd(input_file)
        return k

    def data_preprocess(self):
        k = initial.seperate()
        initial.init_cnndata()
        return k

    def tokenize(self, input_data=None ,ngram_size=None):
        pass

    def load_embedding(self, vec_dir):
        k = initial.init_batchdata(vec_dir)
        return k 
    
    def build_model():
        pass
	
    def train(self, train_data=None):
        print('train model')
        cnn_rnn_model.train(AtNRE_dir +'/cnndata/cnn_train_word.npy', AtNRE_dir +'/cnndata/cnn_train_pos1.npy', AtNRE_dir + '/cnndata/cnn_train_pos2.npy',AtNRE_dir +'/cnndata/cnn_train_y.npy',AtNRE_dir +'/model/best_cnn_model.ckpt')
        pass

    def predict(self, test_data=None, entity_1=None, entity_2=None,  trained_model = None):
        num_total = tester.produce_label_data()
        tester.produce_pred_data(save_path=AtNRE_dir+ '/model/best_cnn_model.ckpt',output_path = AtNRE_dir +'/result/best_pred_entitypair.pkl')
        result = tester.P_N(label_path = AtNRE_dir +'/data/label_entitypair.pkl',pred_path = AtNRE_dir+'/result/best_pred_entitypair.pkl')
        #print ('best_cnn_P@100,200,300:',result)
        List_Precision = []
        List_Recall = []
        Precision, Recall = tester.PR_curve(label_path =AtNRE_dir + '/data/label_entitypair.pkl',pred_path = AtNRE_dir +'/result/best_pred_entitypair.pkl',
                             num_total=num_total)
        List_Precision.append(Precision)
        List_Recall.append(Recall)
        tester.save_pr(List_Precision, List_Recall)
        return List_Precision, List_Recall
        

    def evaluate(self, input_data=None, trained_model = None):
        plot_pr.plot()
        pass

if __name__ == "__main__":
    obj = AtNRE()
    directory = AtNRE_dir + '/origin_data/train.txt'
    vec_dir = AtNRE_dir + '/origin_data/vec.txt'
    obj.read_dataset(directory)
    obj.load_embedding(vec_dir)
    obj.data_preprocess()
    obj.train()
    obj.predict()
    obj.evaluate()
    
		
