#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import Cnn
from model.test import test
from utils.DataManager import DataManager
from utils.Sentence import Sentence

def predict(dataobject,relationdict,**kwargs):
    checkpoint_dir = kwargs.get("model_path", "runs/model_output/checkpoints/")
    embedding_dim = kwargs.get('embedding_dim', 50)
    sequence_length = kwargs.get("sequence_length", 100)
    filter_sizes = kwargs.get('filter_sizes', "3")
    num_filters = kwargs.get("num_filters", 128)
    dropout_keep_prob = kwargs.get("dropout_keep_prob", 0.5)
    l2_reg_lambda = kwargs.get("l2_reg_lambda", 0.0)
    batch_size = kwargs.get("batch_size", 64)
    num_epochs = kwargs.get("num_epochs", 100)
    evaluate_every = kwargs.get("evaluate_every", 1000)
    checkpoint_every = kwargs.get("checkpoint_every", 100)
    allow_soft_placement = kwargs.get("allow_soft_placement", True)
    log_device_placement = kwargs.get("log_device_placement", False)

    # Data Preparation
    # ====================
    datamanager = DataManager(sequence_length,relationdict)
    datamanager.load_relations()
    testing_data = datamanager.load_testing_data(dataobject)
    # test=Sentence(testing_data)
    # print("testing data is",testing_data.relation)
    print("\Predicting...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
            input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            s = graph.get_operation_by_name("output/scores").outputs[0]
            p = graph.get_operation_by_name("output/predictions").outputs[0]

            output_file='predictions.txt'
            results = []
            total = 0
            sequence_length=100
            i = 0
            t = 0
            c = 0
            j=0
            prediction_file = open(output_file, 'w')
            for test in testing_data:
                i += 1
                x_test = datamanager.generate_x(testing_data[test])
                p1, p2 = datamanager.generate_p(testing_data[test])
                y_test = datamanager.generate_y(testing_data[test])
                scores, pre = sess.run([s, p], {input_x: x_test, input_p1:p1, input_p2:p2, dropout_keep_prob: 1.0})
                max_pro = 0
                prediction = -1
                for score in scores:
                    score = np.exp(score-np.max(score))
                    score = score/score.sum(axis=0)
                    score[0] = 0
                    pro = score[np.argmax(score)]
                    if pro > max_pro and np.argmax(score)!=0:
                        max_pro = pro
                        prediction = np.argmax(score)
                for i in range(len(testing_data[test])):
                    results.append((test, testing_data[test][i].relation.id, max_pro, prediction))
                    dm = DataManager(sequence_length,relationdict)
                    relation_data = list(open("data/relation2id.txt", encoding="utf8").readlines())
                    relation_data = [s.split() for s in relation_data]
                    for relation in relation_data:
                        if int(relation[1]) == testing_data[test][i].relation.id:
                            print("Predicted relation is",relation[0])
                            prelation=relation[0]

                    if testing_data[test][i].relation.id == pre and pre!=0:
                        c += 1
                    t += 1
                    if testing_data[test][i].relation.id != 0:
                        total += 1
                list1=dataobject[j].split()
                j=j+1
                prediction_file.writelines("\t".join([' '.join(list1[5:-1]),list1[2],list1[3],list1[4],prelation+'\n']))
                print("\t".join([' '.join(list1[5:-1]),list1[2],list1[3],list1[4],prelation+'\n']))
    return output_file
