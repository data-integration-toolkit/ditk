import sys
import argparse
import math
import os.path
import timeit
import psutil
from graph_embedding import GraphEmbedding
from multiprocessing import JoinableQueue, Queue, Process

import numpy as np
import tensorflow as tf

class ProjE(GraphEmbedding):
    __n_entity=""
    __train_triple=""
    __trainable=""
    __hr_t=""

    __train_hr_t=""
    __train_tr_h=""
    __tr_h=""
    __ent_embedding=""
    __rel_embedding=""
    @property
    def n_entity(self):
        return self.__n_entity

    @property
    def n_train(self):
        return self.__train_triple.shape[0]

    @property
    def trainable_variables(self):
        return self.__trainable

    @property
    def hr_t(self):
        return self.__hr_t

    @property
    def train_hr_t(self):
        return self.__train_hr_t

    @property
    def train_tr_h(self):
        return self.__train_tr_h

    @property
    def tr_h(self):
        return self.__tr_h

    @property
    def ent_embedding(self):
        return self.__ent_embedding

    @property
    def rel_embedding(self):
        return self.__rel_embedding

    def training_data(self, batch_size=100):

        n_triple = len(self.__train_triple)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            hr_tlist, hr_tweight, tr_hlist, tr_hweight = self.corrupted_training(
                self.__train_triple[rand_idx[start:end]])
            yield hr_tlist, hr_tweight, tr_hlist, tr_hweight
            start = end

    def raw_training_data(self, batch_size=100):
        n_triple = len(self.__train_triple)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__train_triple[rand_idx[start:end]]
            start = end

    def testing_data(self, batch_size=100):
        n_triple = len(self.__test_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__test_triple[start:end, :]
            start = end

    def validation_data(self, batch_size=100):
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end

    def writeOutput(self, ent_embed, rel_embed, args):
        output = open("embeddings.txt", "a")
        file = open(os.path.join(args['data_dir'], "entity2id.txt"), "r")
        ent_list = ent_embed.tolist()
        for each in ent_list:
            str1 = file.readline()
            str2 = str1.strip().split()
            output.write("Entity-"+str2[1]+" "+str(each)+"\n")
        file = open(os.path.join(args['data_dir'], "relation2id.txt"), "r")
        rel_list = rel_embed.tolist()
        for each in rel_list:
            str1 = file.readline()
            str2 = str1.strip().split()
            output.write("Relation-"+str2[1]+" "+str(each)+"\n")
        output.close()

    def corrupted_training(self, htr):
        # [head(tail), relation, #of_total_positive_candidates, positive_instances..., negative_instances...]
        hr_tlist = list()
        hr_tweight = list()
        tr_hlist = list()
        tr_hweight = list()

        for idx in range(htr.shape[0]):
            if np.random.uniform(-1, 1) > 0:  # t r predict h
                tr_hweight.append(
                    [1. if x in self.__tr_h[htr[idx, 1]][htr[idx, 2]] else 0. for x in range(self.__n_entity)])
                tr_hlist.append([htr[idx, 1], htr[idx, 2]])
            else:  # h r predict t
                hr_tweight.append(
                    [1. if x in self.__hr_t[htr[idx, 0]][htr[idx, 2]] else 0. for x in range(self.__n_entity)])

                hr_tlist.append([htr[idx, 0], htr[idx, 2]])

        return np.asarray(hr_tlist, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32), \
               np.asarray(tr_hlist, dtype=np.int32), np.asarray(tr_hweight, dtype=np.float32)

    def __init__(self, data_dir="", embed_dim=200, combination_method='simple', dropout=0.5, neg_weight=0.5):
        self.__n_entity=""
        self.__train_triple=""
        self.__trainable=""
        self.__hr_t=""

        self.__train_hr_t=""
        self.__train_tr_h=""
        self.__tr_h=""
        self.__ent_embedding=""
        self.__rel_embedding=""
        self.train_hrt_input = ""
        self.train_hrt_weight = ""
        self.train_trh_input = ""
        self.train_trh_weight = ""
        self.train_loss = ""
        self.train_op = ""
        self.test_input = ""
        self.test_head = ""
        self.test_tail = ""
        self.ent_embeddings = ""
        self.rel_embeddings = ""
        self.session1 = tf.Session()
        self.loadFlag = 0
        self.dict = {}
        if combination_method.lower() not in ['simple', 'matrix']:
            raise NotImplementedError("ProjE does not support using %s as combination method." % combination_method)

        self.__combination_method = combination_method

        self.__embed_dim = embed_dim
        self.__initialized = False

        self.__trainable = list()
        self.__dropout = dropout

        

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor * tf.abs(weights), 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp * tf.abs(weights), 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * tf.abs(weights)  # all ignored elements will have a prob of 0.

    def train(self, inputs, regularizer_weight=1., scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()
            rel_embedding = self.__rel_embedding
            normalized_ent_embedding = self.__ent_embedding

            hr_tlist, hr_tlist_weight, tr_hlist, tr_hlist_weight = inputs

            # (?, dim)
            hr_tlist_h = tf.nn.embedding_lookup(normalized_ent_embedding, hr_tlist[:, 0])
            hr_tlist_r = tf.nn.embedding_lookup(rel_embedding, hr_tlist[:, 1])
            # (?, dim)
            tr_hlist_t = tf.nn.embedding_lookup(normalized_ent_embedding, tr_hlist[:, 0])
            tr_hlist_r = tf.nn.embedding_lookup(rel_embedding, tr_hlist[:, 1])

            if self.__combination_method.lower() == 'simple':
                # shape (?, dim)
                hr_tlist_hr = hr_tlist_h * self.__hr_weighted_vector[
                                           :self.__embed_dim] + hr_tlist_r * self.__hr_weighted_vector[
                                                                             self.__embed_dim:]

                hrt_res = tf.matmul(tf.nn.dropout(tf.tanh(hr_tlist_hr + self.__hr_combination_bias), self.__dropout),
                                    self.__ent_embedding,
                                    transpose_b=True)

                tr_hlist_tr = tr_hlist_t * self.__tr_weighted_vector[
                                           :self.__embed_dim] + tr_hlist_r * self.__tr_weighted_vector[
                                                                             self.__embed_dim:]

                trh_res = tf.matmul(tf.nn.dropout(tf.tanh(tr_hlist_tr + self.__tr_combination_bias), self.__dropout),
                                    self.__ent_embedding,
                                    transpose_b=True)

                self.regularizer_loss = regularizer_loss = tf.reduce_sum(
                    tf.abs(self.__hr_weighted_vector)) + tf.reduce_sum(tf.abs(
                    self.__tr_weighted_vector)) + tf.reduce_sum(tf.abs(self.__ent_embedding)) + tf.reduce_sum(
                    tf.abs(self.__rel_embedding))

            else:

                hr_tlist_hr = tf.nn.dropout(tf.tanh(tf.matmul(tf.concat(1, [hr_tlist_h, hr_tlist_r]),
                                                              self.__hr_combination_matrix) + self.__hr_combination_bias),
                                            self.__dropout)

                hrt_res = tf.matmul(hr_tlist_hr, self.__ent_embedding, transpose_b=True)

                tr_hlist_tr = tf.nn.dropout(tf.tanh(tf.matmul(tf.concat(1, [tr_hlist_t, tr_hlist_r]),
                                                              self.__tr_combination_matrix) + self.__tr_combination_bias),
                                            self.__dropout)

                trh_res = tf.matmul(tr_hlist_tr, self.__ent_embedding, transpose_b=True)

                self.regularizer_loss = regularizer_loss = tf.reduce_sum(
                    tf.abs(self.__hr_combination_matrix)) + tf.reduce_sum(tf.abs(
                    self.__tr_combination_matrix)) + tf.reduce_sum(tf.abs(self.__ent_embedding)) + tf.reduce_sum(
                    tf.abs(self.__rel_embedding))

            self.hrt_softmax = hrt_res_softmax = self.sampled_softmax(hrt_res, hr_tlist_weight)

            hrt_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(hrt_res_softmax, 1e-10, 1.0)) * tf.maximum(0.,
                                                                                   hr_tlist_weight))

            self.trh_softmax = trh_res_softmax = self.sampled_softmax(trh_res, tr_hlist_weight)
            trh_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(trh_res_softmax, 1e-10, 1.0)) * tf.maximum(0., tr_hlist_weight))
            return hrt_loss + trh_loss + regularizer_loss * regularizer_weight

    def test(self, inputs, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            h = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 0])
            t = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 1])
            r = tf.nn.embedding_lookup(self.__rel_embedding, inputs[:, 2])

            ent_mat = tf.transpose(self.__ent_embedding)

            if self.__combination_method.lower() == 'simple':

                # predict tails
                hr = h * self.__hr_weighted_vector[:self.__embed_dim] + r * self.__hr_weighted_vector[
                                                                            self.__embed_dim:]

                hrt_res = tf.matmul(tf.tanh(hr + self.__hr_combination_bias), ent_mat)
                _, tail_ids = tf.nn.top_k(hrt_res, k=self.__n_entity)

                # predict heads

                tr = t * self.__tr_weighted_vector[:self.__embed_dim] + r * self.__tr_weighted_vector[self.__embed_dim:]

                trh_res = tf.matmul(tf.tanh(tr + self.__tr_combination_bias), ent_mat)
                _, head_ids = tf.nn.top_k(trh_res, k=self.__n_entity)

            else:

                hr = tf.matmul(tf.concat(1, [h, r]), self.__hr_combination_matrix)
                hrt_res = (tf.matmul(tf.tanh(hr + self.__hr_combination_bias), ent_mat))
                _, tail_ids = tf.nn.top_k(hrt_res, k=self.__n_entity)

                tr = tf.matmul(tf.concat(1, [t, r]), self.__tr_combination_matrix)
                trh_res = (tf.matmul(tf.tanh(tr + self.__tr_combination_bias), ent_mat))

                _, head_ids = tf.nn.top_k(trh_res, k=self.__n_entity)

            return head_ids, tail_ids

    def learn_embeddings(self, data = './FB15k/', argDict = {}):
        '''
        Wrapper class for training.
        '''
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        file = os.path.join(os.getcwd(),'embeddings.txt')
        self.train_hrt_input, self.train_hrt_weight, self.train_trh_input, self.train_trh_weight, \
        self.train_loss, self.train_op, self.ent_embeddings, self.rel_embeddings = self.train_ops(argDict, learning_rate=argDict['lr'],
                                     optimizer_str=argDict['optimizer'],
                                     regularizer_weight=argDict['loss_weight'])
        return self.train_hrt_input, self.train_hrt_weight, file, self.train_trh_input, self.train_trh_weight, self.train_loss, self.train_op, self.ent_embeddings, self.rel_embeddings

    def load_model(self, load_dir):
        '''
        Loads an existing tensorflow checkpoint.
        '''
        print("loaded", load_dir)
        with tf.Session() as self.session1:
            tf.global_variables_initializer().run()
            print(str(self.session1), " ", str(os.path.exists(load_dir)))
            saver = tf.train.Saver()
            if load_dir is not None: #and os.path.exists(load_dir):
                    saver.restore(self.session1, load_dir)
                    print("restored", load_dir)
                    self.loadFlag = 1
                    return load_dir

    def save_model(self, args, n_iter, session):
        '''
        Saves a tensorflow checkpoint.
        '''
        saver = tf.train.Saver()
        save_path = saver.save(session, os.path.join(args['save_dir'], "ProjE_" + str(args['prefix']) + "_" + str(n_iter) + ".ckpt"))
        return save_path

    def evaluate(self, data = './FB15k/', args = {}, load_dir = ""):
        '''
        Evaluates the embeddings produced by performing head and tail predictions.
        '''
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()

            iter_offset = 0

            if self.session1 != "" and self.loadFlag == 1:
                with tf.Session() as self.session1:
                    tf.global_variables_initializer().run()
                    iter_offset = int(load_dir.split('.')[-2].split('_')[-1]) + 1
                    print("Load model from %s, iteration %d restored." % (load_dir, iter_offset))
                    self.loadFlag = 0
                    total_inst = self.n_train

                    # training data generator
                    raw_training_data_queue = Queue()
                    training_data_queue = Queue()
                    data_generators = list()
                    for i in range(args['n_generator']):
                        data_generators.append(Process(target=data_generator_func, args=(
                            raw_training_data_queue, training_data_queue, self.train_tr_h, self.train_hr_t, self.n_entity, args['neg_weight'])))
                        data_generators[-1].start()

                    evaluation_queue = JoinableQueue()
                    result_queue = Queue()
                    for i in range(args['n_worker']):
                        worker = Process(target=worker_func, args=(evaluation_queue, result_queue, self.hr_t, self.tr_h))
                        worker.start()

                    for data_func, test_type in zip([self.validation_data, self.testing_data], ['VALID', 'TEST']):
                        accu_mean_rank_h = list()
                        accu_mean_rank_t = list()
                        accu_filtered_mean_rank_h = list()
                        accu_filtered_mean_rank_t = list()

                        evaluation_count = 0

                        for testing_data in data_func(batch_size=args['eval_batch']):
                            head_pred, tail_pred = self.session1.run([self.test_head, self.test_tail],
                                                               {self.test_input: testing_data})

                            evaluation_queue.put((testing_data, head_pred, tail_pred))
                            evaluation_count += 1

                        for i in range(args['n_worker']):
                            evaluation_queue.put(None)

                        print("waiting for worker finishes their work")
                        print(evaluation_queue.qsize())
                        print("all worker stopped.")
                        print(evaluation_count)
                        while evaluation_count > 0:
                            evaluation_count -= 1

                            (mrh, fmrh), (mrt, fmrt) = result_queue.get()
                            accu_mean_rank_h += mrh
                            accu_mean_rank_t += mrt
                            accu_filtered_mean_rank_h += fmrh
                            accu_filtered_mean_rank_t += fmrt

                        print(
                            "[%s] INITIALIZATION [HEAD PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                            (test_type, np.mean(accu_mean_rank_h), np.mean(accu_filtered_mean_rank_h),
                            np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 10),
                            np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10)))

                        print(
                            "[%s] INITIALIZATION [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                            (test_type, np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),
                            np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),
                            np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)))

                    for n_iter in range(iter_offset, args['max_iter']):
                        start_time = timeit.default_timer()
                        accu_loss = 0.
                        accu_re_loss = 0.
                        ninst = 0

                        print("initializing raw training data...")
                        nbatches_count = 0
                        for dat in self.raw_training_data(batch_size=args['batch']):
                            raw_training_data_queue.put(dat)
                            nbatches_count += 1
                        print("raw training data initialized.")

                        while nbatches_count > 0:
                            nbatches_count -= 1

                            hr_tlist, hr_tweight, tr_hlist, tr_hweight = training_data_queue.get()

                            l, rl, _ = self.session1.run(
                                [self.train_loss, self.regularizer_loss, self.train_op], {self.train_hrt_input: hr_tlist,
                                                                                 self.train_hrt_weight: hr_tweight,
                                                                                 self.train_trh_input: tr_hlist,
                                                                                 self.train_trh_weight: tr_hweight})

                            accu_loss += l
                            accu_re_loss += rl
                            ninst += len(hr_tlist) + len(tr_hlist)

                            if ninst % (5000) is not None:
                                print(
                                    '[%d sec](%d/%d) : %.2f -- loss : %.5f rloss: %.5f ' % (
                                        timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                                        l / (len(hr_tlist) + len(tr_hlist)),
                                        args['loss_weight'] * (rl / (len(hr_tlist) + len(tr_hlist)))),
                                    end='\r')
                        print("")
                        print("iter %d avg loss %.5f, time %.3f" % (n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

                        if n_iter % args['save_per'] == 0 or n_iter == args['max_iter'] - 1:
                            save_path = self.save_model(args, n_iter, self.session1)
                            print("Model saved at %s" % save_path)

                        if n_iter % args['eval_per'] == 0 or n_iter == args['max_iter'] - 1:

                            for data_func, test_type in zip([self.validation_data, self.testing_data], ['VALID', 'TEST']):
                                accu_mean_rank_h = list()
                                accu_mean_rank_t = list()
                                accu_filtered_mean_rank_h = list()
                                accu_filtered_mean_rank_t = list()

                                evaluation_count = 0

                                for testing_data in data_func(batch_size=args['eval_batch']):
                                    head_pred, tail_pred = self.session1.run([self.test_head, self.test_tail],
                                                                       {self.test_input: testing_data})

                                    evaluation_queue.put((testing_data, head_pred, tail_pred))
                                    evaluation_count += 1

                                for i in range(args['n_worker']):
                                    evaluation_queue.put(None)

                                print("waiting for worker finishes their work")
                                evaluation_queue.join()
                                print("all worker stopped.")
                                while evaluation_count > 0:
                                    evaluation_count -= 1

                                    (mrh, fmrh), (mrt, fmrt) = result_queue.get()
                                    accu_mean_rank_h += mrh
                                    accu_mean_rank_t += mrt
                                    accu_filtered_mean_rank_h += fmrh
                                    accu_filtered_mean_rank_t += fmrt

                                print(
                                    "[%s] ITER %d [HEAD PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                                    (test_type, n_iter, np.mean(accu_mean_rank_h), np.mean(accu_filtered_mean_rank_h),
                                    np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 10),
                                    np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10)))

                                print(
                                    "[%s] ITER %d [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                                    (test_type, n_iter, np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),
                                    np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),
                                    np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)))
                                if n_iter == args['max_iter']-1 and test_type == 'TEST':
                                    print(self.session1.run(self.__ent_embedding))
                                    self.ent_embeddings = self.session1.run(self.__ent_embedding)
                                    self.rel_embeddings = self.session1.run(self.__rel_embedding)
                                    self.writeOutput(self.ent_embeddings, self.rel_embeddings, args)
                                    self.dict = {}
                                    self.dict['mr'] = np.mean(accu_mean_rank_t)
                                    self.dict['mr_filtered'] = np.mean(accu_filtered_mean_rank_t)
                                    self.dict['hits'] = np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10)
                                    self.dict['hits_filtered'] = np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)
                                    for pid in psutil.pids():
                                        p = psutil.Process(pid)
                                        if p.name() == "python3":
                                            p.terminate()
            else:
                total_inst = self.n_train

                # training data generator
                raw_training_data_queue = Queue()
                training_data_queue = Queue()
                data_generators = list()
                for i in range(args['n_generator']):
                    data_generators.append(Process(target=data_generator_func, args=(
                        raw_training_data_queue, training_data_queue, self.train_tr_h, self.train_hr_t, self.n_entity, args['neg_weight'])))
                    data_generators[-1].start()

                evaluation_queue = JoinableQueue()
                result_queue = Queue()
                for i in range(args['n_worker']):
                    worker = Process(target=worker_func, args=(evaluation_queue, result_queue, self.hr_t, self.tr_h))
                    worker.start()

                for data_func, test_type in zip([self.validation_data, self.testing_data], ['VALID', 'TEST']):
                    accu_mean_rank_h = list()
                    accu_mean_rank_t = list()
                    accu_filtered_mean_rank_h = list()
                    accu_filtered_mean_rank_t = list()

                    evaluation_count = 0

                    for testing_data in data_func(batch_size=args['eval_batch']):
                        head_pred, tail_pred = session.run([self.test_head, self.test_tail],
                                                           {self.test_input: testing_data})

                        evaluation_queue.put((testing_data, head_pred, tail_pred))
                        evaluation_count += 1

                    for i in range(args['n_worker']):
                        evaluation_queue.put(None)

                    print("waiting for worker finishes their work")
                    print(evaluation_queue.qsize())
                    print("all worker stopped.")
                    print(evaluation_count)
                    while evaluation_count > 0:
                        evaluation_count -= 1

                        (mrh, fmrh), (mrt, fmrt) = result_queue.get()
                        accu_mean_rank_h += mrh
                        accu_mean_rank_t += mrt
                        accu_filtered_mean_rank_h += fmrh
                        accu_filtered_mean_rank_t += fmrt

                    print(
                        "[%s] INITIALIZATION [HEAD PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                        (test_type, np.mean(accu_mean_rank_h), np.mean(accu_filtered_mean_rank_h),
                        np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 10),
                        np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10)))

                    print(
                        "[%s] INITIALIZATION [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                        (test_type, np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),
                        np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),
                        np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)))

                for n_iter in range(iter_offset, args['max_iter']):
                    start_time = timeit.default_timer()
                    accu_loss = 0.
                    accu_re_loss = 0.
                    ninst = 0

                    print("initializing raw training data...")
                    nbatches_count = 0
                    for dat in self.raw_training_data(batch_size=args['batch']):
                        raw_training_data_queue.put(dat)
                        nbatches_count += 1
                    print("raw training data initialized.")

                    while nbatches_count > 0:
                        nbatches_count -= 1

                        hr_tlist, hr_tweight, tr_hlist, tr_hweight = training_data_queue.get()

                        l, rl, _ = session.run(
                            [self.train_loss, self.regularizer_loss, self.train_op], {self.train_hrt_input: hr_tlist,
                                                                             self.train_hrt_weight: hr_tweight,
                                                                             self.train_trh_input: tr_hlist,
                                                                             self.train_trh_weight: tr_hweight})

                        accu_loss += l
                        accu_re_loss += rl
                        ninst += len(hr_tlist) + len(tr_hlist)

                        if ninst % (5000) is not None:
                            print(
                                '[%d sec](%d/%d) : %.2f -- loss : %.5f rloss: %.5f ' % (
                                    timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                                    l / (len(hr_tlist) + len(tr_hlist)),
                                    args['loss_weight'] * (rl / (len(hr_tlist) + len(tr_hlist)))),
                                end='\r')
                    print("")
                    print("iter %d avg loss %.5f, time %.3f" % (n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

                    if n_iter % args['save_per'] == 0 or n_iter == args['max_iter'] - 1:
                        save_path = self.save_model(args, n_iter, session)
                        print("Model saved at %s" % save_path)

                    if n_iter % args['eval_per'] == 0 or n_iter == args['max_iter'] - 1:

                        for data_func, test_type in zip([self.validation_data, self.testing_data], ['VALID', 'TEST']):
                            accu_mean_rank_h = list()
                            accu_mean_rank_t = list()
                            accu_filtered_mean_rank_h = list()
                            accu_filtered_mean_rank_t = list()

                            evaluation_count = 0

                            for testing_data in data_func(batch_size=args['eval_batch']):
                                head_pred, tail_pred = session.run([self.test_head, self.test_tail],
                                                                   {self.test_input: testing_data})

                                evaluation_queue.put((testing_data, head_pred, tail_pred))
                                evaluation_count += 1

                            for i in range(args['n_worker']):
                                evaluation_queue.put(None)

                            print("waiting for worker finishes their work")
                            evaluation_queue.join()
                            print("all worker stopped.")
                            while evaluation_count > 0:
                                evaluation_count -= 1

                                (mrh, fmrh), (mrt, fmrt) = result_queue.get()
                                accu_mean_rank_h += mrh
                                accu_mean_rank_t += mrt
                                accu_filtered_mean_rank_h += fmrh
                                accu_filtered_mean_rank_t += fmrt

                            print(
                                "[%s] ITER %d [HEAD PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                                (test_type, n_iter, np.mean(accu_mean_rank_h), np.mean(accu_filtered_mean_rank_h),
                                np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 10),
                                np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10)))

                            print(
                                "[%s] ITER %d [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                                (test_type, n_iter, np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),
                                np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),
                                np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)))
                            if n_iter == args['max_iter']-1 and test_type == 'TEST':
                                self.ent_embeddings = session.run(self.__ent_embedding)
                                self.rel_embeddings = session.run(self.__rel_embedding)
                                self.writeOutput(self.ent_embeddings, self.rel_embeddings, args)
                                self.dict = {}
                                self.dict['mr'] = np.mean(accu_mean_rank_t)
                                self.dict['mr_filtered'] = np.mean(accu_filtered_mean_rank_t)
                                self.dict['hits'] = np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10)
                                self.dict['hits_filtered'] = np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)
                                for pid in psutil.pids():
                                    p = psutil.Process(pid)
                                    if p.name() == "python3":
                                        p.terminate()

    def read_dataset(self, dataPath):
        '''
        Reads the data from file paths for entity IDs, relations IDs, train, valid and test triples.
        '''
        tf.reset_default_graph()
        if 'yago' in dataPath.lower() or 'wiki' in dataPath.lower():
            args = {'batch': 200, 'combination_method': 'simple', 'data_dir': dataPath, 'dim': self.__embed_dim, 'drop_out': 0.5,
                    'eval_batch': 500, 'eval_per': 1, 'load_model': '', 'loss_weight': 1e-05, 'lr': 0.01,
                    'max_iter': 2, 'n_generator': 10, 'n_worker': 3, 'neg_weight': 0.5, 'optimizer': 'adam',
                    'prefix': 'DEFAULT', 'save_dir': './trainFiles/', 'save_per': 1, 'summary_dir': './ProjE_summary/', 'datasetFlag': 1}
        else:
            args = {'batch': 200, 'combination_method': 'simple', 'data_dir': dataPath, 'dim': self.__embed_dim, 'drop_out': 0.5,
                    'eval_batch': 500, 'eval_per': 1, 'load_model': '', 'loss_weight': 1e-05, 'lr': 0.01,
                    'max_iter': 3, 'n_generator': 10, 'n_worker': 3, 'neg_weight': 0.5, 'optimizer': 'adam',
                    'prefix': 'DEFAULT', 'save_dir': './trainFiles/', 'save_per': 1, 'summary_dir': './ProjE_summary/', 'datasetFlag': 0}

        data_dir = args['data_dir']        
        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.__n_entity = len(f.readlines())
            f.close()
        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            if args['datasetFlag'] == 0:
                self.__entity_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
                self.__id_entity_map = {v: k for k, v in self.__entity_id_map.items()}
            else:
                self.__entity_id_map = {x.strip().split('\t')[0]: x.strip().split('\t')[1] for x in f.readlines()}
                self.__id_entity_map = {v: k for k, v in self.__entity_id_map.items()}
            f.close()

        print("N_ENTITY: %d" % self.__n_entity)

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.__n_relation = len(f.readlines())
            f.close()

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            if args['datasetFlag'] == 0:
                self.__relation_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
                self.__id_relation_map = {v: k for k, v in self.__entity_id_map.items()}
            else:
                self.__relation_id_map = {x.strip().split('\t')[0]: x.strip().split('\t')[1] for x in f.readlines()}
                self.__id_relation_map = {v: k for k, v in self.__entity_id_map.items()}
            f.close()

        print("N_RELATION: %d" % self.__n_relation)

        def load_triple(file_path):
            with open(file_path, 'r', encoding='utf-8') as f_triple:
                if args['datasetFlag'] == 0:
                    return np.asarray([[self.__entity_id_map[x.strip().split('\t')[0]],
                                        self.__entity_id_map[x.strip().split('\t')[1]],
                                        self.__relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
                                      dtype=np.int32)
                else:
                    return np.asarray([[x.strip().split('\t')[0],
                                   x.strip().split('\t')[1],
                                   x.strip().split('\t')[2]] for x in f_triple.readlines()],
                                  dtype=np.int32)

        def gen_hr_t(triple_data):
            hr_t = dict()
            for h, t, r in triple_data:
                if h not in hr_t:
                    hr_t[h] = dict()
                if r not in hr_t[h]:
                    hr_t[h][r] = set()
                hr_t[h][r].add(t)

            return hr_t

        def gen_tr_h(triple_data):
            tr_h = dict()
            for h, t, r in triple_data:
                if t not in tr_h:
                    tr_h[t] = dict()
                if r not in tr_h[t]:
                    tr_h[t][r] = set()
                tr_h[t][r].add(h)
            return tr_h

        self.__train_triple = load_triple(os.path.join(data_dir, 'train.txt'))
        if args['datasetFlag'] == 1:
            b = self.__train_triple[:, [0, 2, 1]]
            self.__train_triple = b
        print("N_TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])

        self.__test_triple = load_triple(os.path.join(data_dir, 'test.txt'))
        if args['datasetFlag'] == 1:
            b = self.__test_triple[:, [0, 2, 1]]
            self.__test_triple = b
        print("N_TEST_TRIPLES: %d" % self.__test_triple.shape[0])

        self.__valid_triple = load_triple(os.path.join(data_dir, 'valid.txt'))
        if args['datasetFlag'] == 1:
            b = self.__valid_triple[:, [0, 2, 1]]
            self.__valid_triple = b
        print("N_VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

        self.__train_hr_t = gen_hr_t(self.__train_triple)
        self.__train_tr_h = gen_tr_h(self.__train_triple)
        self.__test_hr_t = gen_hr_t(self.__test_triple)
        self.__test_tr_h = gen_tr_h(self.__test_triple)

        self.__hr_t = gen_hr_t(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))
        self.__tr_h = gen_tr_h(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))

        embed_dim=args['dim']
        bound = 6 / math.sqrt(embed_dim)

        with tf.device('/cpu'):
            self.__ent_embedding = tf.get_variable("ent_embedding", [self.__n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=345))
            self.__trainable.append(self.__ent_embedding)

            self.__rel_embedding = tf.get_variable("rel_embedding", [self.__n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=346))
            self.__trainable.append(self.__rel_embedding)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            self.ent_embeddings = sess.run(self.__ent_embedding)
            self.rel_embeddings = sess.run(self.__rel_embedding)
            combination_method=args['combination_method']
            if combination_method.lower() == 'simple':
                self.__hr_weighted_vector = tf.get_variable("simple_hr_combination_weights", [embed_dim * 2],
                                                            initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                      maxval=bound,
                                                                                                      seed=445))
                self.__tr_weighted_vector = tf.get_variable("simple_tr_combination_weights", [embed_dim * 2],
                                                            initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                      maxval=bound,
                                                                                                      seed=445))
                self.__trainable.append(self.__hr_weighted_vector)
                self.__trainable.append(self.__tr_weighted_vector)
                self.__hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                             initializer=tf.zeros([embed_dim]))
                self.__tr_combination_bias = tf.get_variable("combination_bias_tr",
                                                             initializer=tf.zeros([embed_dim]))

                self.__trainable.append(self.__hr_combination_bias)
                self.__trainable.append(self.__tr_combination_bias)

            else:
                self.__hr_combination_matrix = tf.get_variable("matrix_hr_combination_layer",
                                                               [embed_dim * 2, embed_dim],
                                                               initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                         maxval=bound,
                                                                                                         seed=555))
                self.__tr_combination_matrix = tf.get_variable("matrix_tr_combination_layer",
                                                               [embed_dim * 2, embed_dim],
                                                               initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                         maxval=bound,
                                                                                                         seed=555))
                self.__trainable.append(self.__hr_combination_matrix)
                self.__trainable.append(self.__tr_combination_matrix)
                self.__hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                             initializer=tf.zeros([embed_dim]))
                self.__tr_combination_bias = tf.get_variable("combination_bias_tr",
                                                             initializer=tf.zeros([embed_dim]))

                self.__trainable.append(self.__hr_combination_bias)
                self.__trainable.append(self.__tr_combination_bias)
        file = open(os.path.join(data_dir, 'train.txt'), "r")
        trainList = file.readlines()
        file.close()
        file = open(os.path.join(data_dir, 'valid.txt'), "r")
        validList = file.readlines()
        file.close()
        file = open(os.path.join(data_dir, 'test.txt'), "r")
        testList = file.readlines()
        file.close()
        return args, trainList, validList, testList


    def train_ops(self, args, learning_rate=0.1, optimizer_str='gradient', regularizer_weight=1.0):
        with tf.device('/cpu'):
            self.train_hrt_input = tf.placeholder(tf.int32, [None, 2])
            self.train_hrt_weight = tf.placeholder(tf.float32, [None, self.n_entity])
            self.train_trh_input = tf.placeholder(tf.int32, [None, 2])
            self.train_trh_weight = tf.placeholder(tf.float32, [None, self.n_entity])
    	
            loss = self.train([self.train_hrt_input, self.train_hrt_weight, self.train_trh_input, self.train_trh_weight],
                               regularizer_weight=regularizer_weight)
            if optimizer_str == 'gradient':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif optimizer_str == 'rms':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif optimizer_str == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            else:
                raise NotImplementedError("Does not support %s optimizer" % optimizer_str)

            grads = optimizer.compute_gradients(loss, self.trainable_variables)

            op_train = optimizer.apply_gradients(grads)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            self.ent_embeddings = sess.run(self.__ent_embedding)
            self.rel_embeddings = sess.run(self.__rel_embedding)
            #self.writeOutput(self.ent_embeddings, self.rel_embeddings, args)
            return self.train_hrt_input, self.train_hrt_weight, self.train_trh_input, self.train_trh_weight, loss, op_train, self.ent_embeddings, self.rel_embeddings

    def test_ops(self):
        with tf.device('/cpu'):
            self.test_input = tf.placeholder(tf.int32, [None, 3])
            head_ids, tail_ids = self.test(self.test_input)

        return self.test_input, head_ids, tail_ids


def worker_func(in_queue, out_queue, hr_t, tr_h):
    while True:
        dat = in_queue.get()
        if dat is None:
            in_queue.task_done()
            # break
            continue
        testing_data, head_pred, tail_pred = dat
        out_queue.put(test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h))
        in_queue.task_done()
    print('Done with Worker Fun')


def data_generator_func(in_queue, out_queue, tr_h, hr_t, n_entity, neg_weight):
    while True:
        dat = in_queue.get()
        if dat is None:
            break
        # [head(tail), relation, #of_total_positive_candidates, positive_instances..., negative_instances...]
        hr_tlist = list()
        hr_tweight = list()
        tr_hlist = list()
        tr_hweight = list()

        htr = dat

        for idx in range(htr.shape[0]):
            if np.random.uniform(-1, 1) > 0:  # t r predict h
                tr_hweight.append(
                    [1. if x in tr_h[htr[idx, 1]][htr[idx, 2]] else y for
                     x, y in enumerate(np.random.choice([0., -1.], size=n_entity, p=[1 - neg_weight, neg_weight]))])
                tr_hlist.append([htr[idx, 1], htr[idx, 2]])
            else:  # h r predict t
                hr_tweight.append(
                    [1. if x in hr_t[htr[idx, 0]][htr[idx, 2]] else y for
                     x, y in enumerate(np.random.choice([0., -1.], size=n_entity, p=[1 - neg_weight, neg_weight]))])

                hr_tlist.append([htr[idx, 0], htr[idx, 2]])

        out_queue.put((np.asarray(hr_tlist, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32),
                       np.asarray(tr_hlist, dtype=np.int32), np.asarray(tr_hweight, dtype=np.float32)))
    print('Done with DATA GEN')


def test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h):
    assert len(testing_data) == len(head_pred)
    assert len(testing_data) == len(tail_pred)

    mean_rank_h = list()
    mean_rank_t = list()
    filtered_mean_rank_h = list()
    filtered_mean_rank_t = list()

    for i in range(len(testing_data)):
        h = testing_data[i, 0]
        t = testing_data[i, 1]
        r = testing_data[i, 2]
        # mean rank

        mr = 0
        for val in head_pred[i]:
            if val == h:
                mean_rank_h.append(mr)
                break
            mr += 1

        mr = 0
        for val in tail_pred[i]:
            if val == t:
                mean_rank_t.append(mr)
            mr += 1

        # filtered mean rank
        fmr = 0
        for val in head_pred[i]:
            if val == h:
                filtered_mean_rank_h.append(fmr)
                break
            if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
                continue
            else:
                fmr += 1

        fmr = 0
        for val in tail_pred[i]:
            if val == t:
                filtered_mean_rank_t.append(fmr)
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1

    return (mean_rank_h, filtered_mean_rank_h), (mean_rank_t, filtered_mean_rank_t)
