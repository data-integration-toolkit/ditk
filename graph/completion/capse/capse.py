import tensorflow as tf
from scipy.stats import rankdata

import numpy as np
import os
import time
import datetime
from builddata_softplus import *
from capsuleNet import CapsEBackend

import math
from pprint import pprint

import sys
sys.path.append("..")
from graph_completion import graph_completion


# TODO Add flags to make sure methods are run in proper order

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class CapsE(graph_completion):
    def __init__(self, path):
        np.random.seed(1234)
        tf.set_random_seed(1234)

        # Parameters
        self.data = path
        self.run_folder = "./"
        self.model_index = '2'
        self.embedding_dim = 100
        self.filter_size = 1
        self.num_filters = 400
        self.learning_rate = 0.00001
        self.batch_size = 128
        self.neg_ratio = 1.0
        self.num_epochs = 31
        self.savedEpochs = 30
        self.allow_soft_placement = True
        self.log_device_placement = False
        self.model_name = str(id(self))
        self.iter_routing = 1
        self.num_outputs_secondCaps = 1
        self.vec_len_secondCaps = 10

        self.useConstantInit = False # REVIEW



        print("Loading data... finished!")

    def read_dataset(self, fileName, options={}):
        """
        Reads and returns a dataset

        Args:
            fileName: Name of dataset to read
            options: object to store any extra or implementation specific data

        Returns:
            training, testing, and validation data
        """
        assert(fileName in ["fb15k", "wn18"])

        train, valid, test, self.words_indexes, self.indexes_words, self.headTailSelector, self.entity2id, self.id2entity, self.relation2id, self.id2relation = build_data(fileName, self.data)
        self.data_size = len(train)


        # self.initialization = []
        self.initialization = np.empty([len(self.words_indexes), self.embedding_dim]).astype(np.float32)
        # initEnt, initRel = init_norm_Vector(self.data + self.name + '/relation2vec' + str(self.embedding_dim) + '.init',
        #                                         self.data + self.name + '/entity2vec' + str(self.embedding_dim) + '.init', self.embedding_dim)

        for _word in self.words_indexes:
            if _word in self.relation2id:
                _ind = self.words_indexes[_word]
                self.initialization[_ind] = [random.uniform(-1, 1) for k in range(self.embedding_dim)] #initRel[index]
            elif _word in self.entity2id:
                _ind = self.words_indexes[_word]
                self.initialization[_ind] = [random.uniform(-1, 1) for k in range(self.embedding_dim)] # initEnt[index]
            else:
                print('*****************Error********************!')
                break
        self.initialization = np.array(self.initialization, dtype=np.float32)

        self._train = train
        self._test = test
        self._valid = valid

        assert len(self.words_indexes) % (len(self.entity2id) + len(self.relation2id)) == 0

        return train, test, valid

    def train(self, data, options={}):
        """
        Trains a model on the given input data

        Args:
            data: [(subject, relation, object, ...)]
            options: object to store any extra or implementation specific data

        Returns:
            None. Generated embedding is stored in instance state. 
        """
        self.train_batch = Batch_Loader(data, self.words_indexes, self.indexes_words, self.headTailSelector, \
                                self.entity2id, self.id2entity, self.relation2id, self.id2relation, batch_size=self.batch_size, neg_ratio=self.neg_ratio)
        self.entity_array = np.array(list(self.train_batch.indexes_ents.keys()))

        x_train = np.array(list(data.keys())).astype(np.int32)
        print("[!] Training model...")
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                global_step = tf.Variable(0, name="global_step", trainable=False)
                capse = CapsEBackend(sequence_length=x_train.shape[1], #self.x_valid
                                    initialization=self.initialization,
                                    embedding_size=self.embedding_dim,
                                    filter_size=self.filter_size,
                                    num_filters=self.num_filters,
                                    vocab_size=len(self.words_indexes),
                                    iter_routing=self.iter_routing,
                                    batch_size=2*self.batch_size,
                                    num_outputs_secondCaps=self.num_outputs_secondCaps,
                                    vec_len_secondCaps=self.vec_len_secondCaps,
                                    useConstantInit=self.useConstantInit)

                # Define Training procedure
                #optimizer = tf.contrib.opt.NadamOptimizer(1e-3)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                grads_and_vars = optimizer.compute_gradients(capse.total_loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                out_dir = os.path.abspath(os.path.join(self.run_folder, "runs_CapsE", self.model_name))

                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        capse.input_x: x_batch,
                        capse.input_y: y_batch
                    }
                    _, step, loss = sess.run([train_op, global_step, capse.total_loss], feed_dict)
                    return loss

                num_batches_per_epoch = int((self.data_size - 1) / self.batch_size) + 1
                for epoch in range(self.num_epochs):
                    print("Epoch:", epoch)
                    for batch_num in range(num_batches_per_epoch):
                        x_batch, y_batch = self.train_batch()
                        loss = train_step(x_batch, y_batch)
                        current_step = tf.train.global_step(sess, global_step)               
                    if epoch > 0:
                        if epoch % self.savedEpochs == 0:
                            path = capse.saver.save(sess, checkpoint_prefix, global_step=int(self.model_index))
                    print("Loss in iteration", epoch, "=", loss)

    def predict(self, data, options={}):
        """
        Use generated embedding to predicts links the given input data (KG as list of triples).
        Assumes embedding has been generated model via train()

        Args:
            data: [(subject, relation, object, ...)]
            options: object to store any extra or implementation specific data

        Returns:
            predictions: [tuple,...], i.e. list of predicted tuples. 
                Each tuple will follow format: (subject_entity, relation, object_entity)
        """
        self.x_test = np.array(list(data.keys())).astype(np.int32)
        self.y_test = np.array(list(data.values())).astype(np.float32)
        
        num_splits = 8
        

        len_test = len(self.x_test)
        batch_test = int(len_test / (num_splits - 1))

        print("[!] Predicting links...")
        with tf.Graph().as_default():
            tf.set_random_seed(1234)
            session_conf = tf.ConfigProto(allow_soft_placement=self.allow_soft_placement,
                                        log_device_placement=self.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                global_step = tf.Variable(0, name="global_step", trainable=False)

                capse = CapsEBackend(sequence_length=self.x_test.shape[1],
                            initialization=self.initialization,
                            embedding_size=self.embedding_dim,
                            filter_size=self.filter_size,
                            num_filters=self.num_filters,
                            vocab_size=len(self.words_indexes),
                            iter_routing=self.iter_routing,
                            batch_size=2 * self.batch_size,
                            num_outputs_secondCaps=self.num_outputs_secondCaps,
                            vec_len_secondCaps=self.vec_len_secondCaps,
                            useConstantInit=self.useConstantInit
                            )
                out_dir = os.path.abspath(os.path.join(self.run_folder, "runs_CapsE", self.model_name))

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")

                lstModelIndexes = list(self.model_index.split(","))

                for _model_index in lstModelIndexes:

                    _file = checkpoint_prefix + "-" + _model_index

                    capse.saver.restore(sess, _file)


                    # Predict function to predict scores for test data
                    def predict(x_batch, y_batch, writer=None):
                        feed_dict = {
                            capse.input_x: x_batch,
                            capse.input_y: y_batch
                        }
                        scores = sess.run([capse.predictions], feed_dict)
                        return scores


                    def test_prediction(x_batch, y_batch, head_or_tail='head'):

                        hits10 = 0.0
                        mrr = 0.0
                        mr = 0.0
                        hits1 = 0.0
                        

                        newts = set()
                        for i in range(len(x_batch)):
                            new_x_batch = np.tile(x_batch[i], (len(self.entity2id), 1))
                            new_y_batch = np.tile(y_batch[i], (len(self.entity2id), 1))


                            if head_or_tail == 'head':
                                new_x_batch[:, 0] = self.entity_array
                            else:  # 'tail'
                                new_x_batch[:, 2] = self.entity_array

                            lstIdx = []
                            for tmpIdxTriple in range(len(new_x_batch)):
                                tmpTriple = (new_x_batch[tmpIdxTriple][0], new_x_batch[tmpIdxTriple][1],
                                                new_x_batch[tmpIdxTriple][2])
                                if (tmpTriple in self._train) or (tmpTriple in self._valid) or (
                                        tmpTriple in self._test):  # also remove the valid test triple
                                    lstIdx.append(tmpIdxTriple)
                            new_x_batch = np.delete(new_x_batch, lstIdx, axis=0)
                            new_y_batch = np.delete(new_y_batch, lstIdx, axis=0)

                            # thus, insert the valid test triple again, to the beginning of the array
                            new_x_batch = np.insert(new_x_batch, 0, x_batch[i],
                                                    axis=0)  # thus, the index of the valid test triple is equal to 0
                            new_y_batch = np.insert(new_y_batch, 0, y_batch[i], axis=0)

                            #for running with a batch size
                            while len(new_x_batch) % ((int(self.neg_ratio) + 1) * self.batch_size) != 0:
                                new_x_batch = np.append(new_x_batch, [x_batch[i]], axis=0)
                                new_y_batch = np.append(new_y_batch, [y_batch[i]], axis=0)

                            results = []
                            listIndexes = range(0, len(new_x_batch), (int(self.neg_ratio) + 1) * self.batch_size)
                            for tmpIndex in range(len(listIndexes) - 1):
                                results = np.append(results, predict(
                                    new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
                                    new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]]))

                           

                            results = np.append(results,
                                                predict(new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:]))
                            

                            results = np.reshape(results, -1)

                            results_with_id = rankdata(results, method='ordinal')
                            _filter = results_with_id[0]


                            threshold = .55 
                            for j, r in enumerate(results):
                                if r <= threshold and r != results[-1]:
                                    newts.add(tuple(new_x_batch[j].tolist()))

                            mr += _filter
                            mrr += 1.0 / _filter
                            if _filter <= 10:
                                hits10 += 1
                            
                            if _filter == 1:
                                hits1 += 1

                        return np.array([mr/i, mrr/i, hits1/i, hits10/i]), newts

                    # REVIEW: testIdx => [0,7]
                    testIdx = 1
                   
                    h_results, new_trips_h = test_prediction(self.x_test, self.y_test, head_or_tail="head")
                    t_results, new_trips_t = test_prediction(self.x_test, self.y_test, head_or_tail="tail")

                    self.evals = {
                        "mr": (h_results[0] + t_results[0]) / 2,
                        "mrr": (h_results[1] + t_results[1]) / 2,
                        "hits1": (h_results[2] + t_results[2]) / 2,
                        "hits10": (h_results[3] + t_results[3]) / 2
                    }

                    return new_trips_h.union(new_trips_t)

    def evaluate(self, benchmark_data, metrics={}, options={}):
        """
        Calculates evaluation metrics on chosen benchmark dataset.
        Precondition: model has been trained and predictions were generated from predict()

        Args:
            benchmark_data: [(subject, relation, object, ...)]
            metrics: List of metrics for desired evaluation metrics (e.g. hits1, hits10, mrr)
            options: object to store any extra or implementation specific data

        Returns:
            evaluations: dictionary of scores with respect to chosen metrics
                - e.g.
                    evaluations = {
                        "hits10": 0.5,
                        "mrr": 0.8
                    }
        """
        print("[!] Evaluating model...")
        self.predict(benchmark_data)
        evals = {m:self.evals[m] for m in metrics}
        if len(evals) != len(metrics):
            print("Please enter valid metrics: {mrr, hits1, hits10}")
        
        return evals


if __name__ == "__main__":
    c = CapsE("../data")
    datasets = ["fb15k", "wn18"]
    training, test, valid = c.read_dataset(datasets[0])
    c.train(training)
    new_triples = c.predict(test)
    print("{0} new triples generated".format(len(new_triples)))
    print(c.evaluate(valid, ["mrr", "hits10"]))