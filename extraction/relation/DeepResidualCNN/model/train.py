import tensorflow as tf
import numpy as np
import os
import time
import datetime
import Cnn
from model.test import test
from utils.DataManager import DataManager
import warnings
warnings.filterwarnings('ignore')


def train(dataobject,relationdict,**kwargs):
    relationdict=relationdict
    print("relations",relationdict)
    sequence_length=kwargs.get('sequence_length',100)
    embedding_dim=kwargs.get('embedding_dim',50)
    filter_sizes = kwargs.get('filter_sizes', "3")
    num_filters = kwargs.get("num_filters", 128)
    dropout_keep_prob = kwargs.get("dropout_keep_prob", 0.5)
    l2_reg_lambda = kwargs.get("l2_reg_lambda", 0.0)
    batch_size = kwargs.get("batch_size", 64)
    num_epochs = kwargs.get("num_epochs", 5)
    evaluate_every = kwargs.get("evaluate_every", 1000)
    checkpoint_every = kwargs.get("checkpoint_every", 100)
    allow_soft_placement = kwargs.get("allow_soft_placement", True)
    log_device_placement = kwargs.get("log_device_placement", False)
    model_path=kwargs.get("model_path","model_checkpoint")

    datamanager = DataManager(sequence_length,relationdict)
    datamanager.load_relations()
    training_data = datamanager.load_training_data(dataobject)
    training_data = np.array(training_data)
    testing_data = datamanager.load_testing_data(dataobject)

    # Random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(training_data)))
    training_data = training_data[shuffle_indices]
    print("Finish randomize data")

    train = training_data
    dev = training_data[-1000:]

    # Start Training
    # ====================
    print("Start Training")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = Cnn.RECnn(
                sequence_length,
                len(datamanager.relations),
                embedding_dim,
                5,
                list(map(int, filter_sizes.split(","))),
                num_filters,
                l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_path))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            print("Initialize variables.")
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, p1_batch, p2_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.input_p1: p1_batch,
                    cnn.input_p2: p2_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                return loss

            def dev_step(x_batch, y_batch, p1_batch, p2_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.input_p1: p1_batch,
                    cnn.input_p2: p2_batch,
                    cnn.dropout_keep_prob: 1
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

            # Generate batches
            batches = datamanager.batch_iter(
                train, batch_size, num_epochs)
            num_batches_per_epoch = int(len(train) / batch_size) + 1
            print("Batch data")
            # Training loop. For each batch...
            num_batch = 1
            num_epoch = 1
            dev_x_batch = datamanager.generate_x(dev)
            dev_p1_batch, dev_p2_batch = datamanager.generate_p(dev)
            dev_y_batch = datamanager.generate_y(dev)
            for batch in batches:
                if num_batch == num_batches_per_epoch:
                    num_epoch += 1
                    num_batch = 1
                    test(testing_data, cnn.input_x, cnn.input_p1, cnn.input_p2, cnn.scores, cnn.predictions,
                         cnn.dropout_keep_prob, datamanager, sess, num_epoch,relationdict,dataobject)
                num_batch += 1
                x_batch = datamanager.generate_x(batch)
                p1_batch, p2_batch = datamanager.generate_p(batch)
                y_batch = datamanager.generate_y(batch)
                loss = train_step(x_batch, y_batch, p1_batch, p2_batch)
                current_step = tf.train.global_step(sess, global_step)
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                if current_step % evaluate_every == 0:
                    print("Num_batch: {}".format(num_batch))
                    print("Num_epoch: {}".format(num_epoch))
                    dev_step(dev_x_batch, dev_y_batch, dev_p1_batch, dev_p2_batch)