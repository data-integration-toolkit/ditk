#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import Cnn
from model.test import test
from utils.DataManager import DataManager

def eval(dataobject,relationdict,**kwargs):
    checkpoint_dir = kwargs.get("model_path", "runs/model_checkpoint/checkpoints/")
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

    datamanager = DataManager(sequence_length,relationdict)
    datamanager.load_relations()
    testing_data = datamanager.load_testing_data(dataobject)

    print("\nEvaluating...\n")

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
           

            test(testing_data, input_x, input_p1, input_p2, s, p, dropout_keep_prob, datamanager, sess, -1,relationdict,dataobject)
            
def load_model(**kwargs):
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
    return graph,sess
    
