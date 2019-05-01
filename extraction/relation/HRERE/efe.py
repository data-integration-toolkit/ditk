from model import Model
import tensorflow as tf
import datetime
import numpy as np
import config
from utils.tf_utils import l2_loss, l1_loss

class TransE_L2(Model):
    def __init__(self, n_entities, n_relations, hparams):
        super(TransE_L2, self).__init__(n_entities, n_relations, hparams)
        self.margin = hparams.margin
        self.build()

    def add_params(self):
        minVal = -6 / np.sqrt(self.embedding_size)
        maxVal = -minVal
        self.entity_embedding = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], minVal, maxVal, seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding")
        self.relation_embedding = tf.Variable(tf.nn.l2_normalize(tf.random_uniform(
            [self.n_relations, self.embedding_size], minVal, maxVal, seed=config.RANDOM_SEED), -1),
            dtype=tf.float32, name="relation_embedding")

        normalized_entity_embedding = tf.nn.l2_normalize(self.entity_embedding, -1)
        self.normalization = self.entity_embedding.assign(normalized_entity_embedding)

    def add_prediction_op(self):
        self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
        self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)
        self.r = tf.nn.embedding_lookup(self.relation_embedding, self.relations)

        self.pred = tf.negative(l2_loss(self.e1 + self.r - self.e2), name="pred")

    def add_loss_op(self):
        pos_size, neg_size = self.batch_size, self.batch_size * self.neg_ratio * 2
        score_pos, score_neg = tf.split(self.pred, [pos_size, neg_size])

        losses = tf.maximum(0.0, self.margin - score_pos +
            tf.reduce_mean(tf.reshape(score_neg, (self.batch_size, self.neg_ratio * 2)), -1))
        self.loss = tf.reduce_mean(losses, name="loss")

    def train_on_batch(self, sess, input_batch):
        feed = self.create_feed_dict(**input_batch)
        sess.run(self.normalization)
        _, step, loss = sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))

class TransE_L1(TransE_L2):
    def add_prediction_op(self):
        self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
        self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)
        self.r = tf.nn.embedding_lookup(self.relation_embedding, self.relations)

        self.pred = tf.negative(l1_loss(self.e1 + self.r - self.e2), name="pred")

    def add_loss_op(self):
        pos_size, neg_size = self.batch_size, self.batch_size * self.neg_ratio * 2
        score_pos, score_neg = tf.split(self.pred, [pos_size, neg_size])

        losses = tf.maximum(0.0, self.margin - score_pos +
            tf.reduce_mean(tf.reshape(score_neg, (self.batch_size, self.neg_ratio * 2)), -1))
        self.loss = tf.reduce_mean(losses, name="loss")

class DistMult(Model):
    def __init__(self, n_entities, n_relations, hparams):
        super(DistMult, self).__init__(n_entities, n_relations, hparams)
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.build()

    def add_params(self):
        self.entity_embedding = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding")
        self.relation_embedding = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding")

    def add_prediction_op(self):
        self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
        self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)
        self.r = tf.nn.embedding_lookup(self.relation_embedding, self.relations)

        self.pred = tf.nn.sigmoid(tf.reduce_sum(self.e1 * self.r * self.e2, -1), name="pred")

    def add_loss_op(self):
        losses = tf.nn.softplus(-self.labels * tf.reduce_sum(self.e1 * self.r * self.e2, -1))
        self.l2_loss = tf.reduce_mean(tf.square(self.e1)) + \
            tf.reduce_mean(tf.square(self.e2)) + \
            tf.reduce_mean(tf.square(self.r))
        self.loss = tf.add(tf.reduce_mean(losses), self.l2_reg_lambda * self.l2_loss, name="loss")

class DistMult_tanh(DistMult):
    def add_prediction_op(self):
        self.e1 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding, self.heads))
        self.e2 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding, self.tails))
        self.r = tf.nn.embedding_lookup(self.relation_embedding, self.relations)

        self.pred = tf.nn.sigmoid(tf.reduce_sum(self.e1 * self.r * self.e2, -1), name="pred")

class NTN(Model):
    def __init__(self, n_entities, n_relations, hparams):
        super(NTN, self).__init__(n_entities, n_relations, hparams)
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.k = hparams.k
        self.build()

    def add_params(self):
        self.entity_embedding = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding")
        self.W = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k, self.embedding_size, self.embedding_size],
            0., 1., seed=config.RANDOM_SEED), dtype=tf.float32, name="W")
        self.V = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k, 2 * self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="V")
        self.B = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="B")
        self.U = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="U")

    def add_prediction_op(self):
        self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
        self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)
        self.w = tf.nn.embedding_lookup(self.W, self.relations)
        self.v = tf.nn.embedding_lookup(self.V, self.relations)
        self.b = tf.nn.embedding_lookup(self.B, self.relations)
        self.u = tf.nn.embedding_lookup(self.U, self.relations)

        g_a = tf.matmul(self.v, tf.expand_dims(tf.concat([self.e1, self.e2], 1), -1))
        e1 = tf.reshape(self.e1, [-1, 1, 1, self.embedding_size])
        e2 = tf.reshape(self.e2, [-1, 1, self.embedding_size, 1])
        e1 = tf.tile(e1, [1, self.k, 1, 1])
        e2 = tf.tile(e2, [1, self.k, 1, 1])
        g_b = tf.squeeze(tf.matmul(tf.matmul(e1, self.w), e2), -1)
        b = tf.expand_dims(self.b, -1)

        self.score = tf.squeeze(tf.matmul(tf.expand_dims(self.u, 1), tf.nn.tanh(g_a + g_b + b)))
        self.pred = tf.nn.sigmoid(self.score, name="pred")

    def add_loss_op(self):
        losses = tf.nn.softplus(-self.labels * self.score)
        self.l2_loss = tf.reduce_mean(tf.square(self.e1)) \
            + tf.reduce_mean(tf.square(self.e2)) \
            + tf.reduce_mean(tf.square(self.w)) \
            + tf.reduce_mean(tf.square(self.v)) \
            + tf.reduce_mean(tf.square(self.b)) \
            + tf.reduce_mean(tf.square(self.u))
        self.loss = tf.add(tf.reduce_mean(losses), self.l2_reg_lambda * self.l2_loss, name="loss")

class NTN_diag(NTN):
    def add_params(self):
        self.entity_embedding = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding")
        self.W = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="W")
        self.V = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k, 2 * self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="V")
        self.B = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="B")
        self.U = tf.Variable(tf.random_uniform(
            [self.n_relations, self.k], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="U")

    def add_prediction_op(self):
        self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
        self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)
        self.w = tf.nn.embedding_lookup(self.W, self.relations)
        self.v = tf.nn.embedding_lookup(self.V, self.relations)
        self.b = tf.nn.embedding_lookup(self.B, self.relations)
        self.u = tf.nn.embedding_lookup(self.U, self.relations)

        g_a = tf.matmul(self.v, tf.expand_dims(tf.concat([self.e1, self.e2], 1), -1))
        e1 = tf.reshape(self.e1, [-1, 1, self.embedding_size])
        e2 = tf.reshape(self.e2, [-1, 1, self.embedding_size])
        e1 = tf.tile(e1, [1, self.k, 1])
        e2 = tf.tile(e2, [1, self.k, 1])
        g_b = tf.reduce_sum(e1 * self.w * e2, -1, keep_dims=True)
        b = tf.expand_dims(self.b, -1)

        self.score = tf.squeeze(tf.matmul(tf.expand_dims(self.u, 1), tf.nn.tanh(g_a + g_b + b)))
        self.pred = tf.nn.sigmoid(self.score, name="pred")

class Complex(Model):
    def __init__(self, n_entities, n_relations, hparams):
        super(Complex, self).__init__(n_entities, n_relations, hparams)
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.build()

    def add_params(self):
        self.entity_embedding1 = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding1")
        self.entity_embedding2 = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding2")
        self.relation_embedding1 = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding1")
        self.relation_embedding2 = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding2")

    def add_prediction_op(self):
        self.e1_1 = tf.nn.embedding_lookup(self.entity_embedding1, self.heads)
        self.e1_2 = tf.nn.embedding_lookup(self.entity_embedding2, self.heads)
        self.e2_1 = tf.nn.embedding_lookup(self.entity_embedding1, self.tails)
        self.e2_2 = tf.nn.embedding_lookup(self.entity_embedding2, self.tails)
        self.r_1 = tf.nn.embedding_lookup(self.relation_embedding1, self.relations)
        self.r_2 = tf.nn.embedding_lookup(self.relation_embedding2, self.relations)

        self.pred = tf.subtract(tf.reduce_sum(self.e1_1 * self.r_1 * self.e2_1, -1) +
                                tf.reduce_sum(self.e1_2 * self.r_1 * self.e2_2, -1) +
                                tf.reduce_sum(self.e1_1 * self.r_2 * self.e2_2, -1),
                                tf.reduce_sum(self.e1_2 * self.r_2 * self.e2_1, -1), name="pred")

    def add_loss_op(self):
        losses = tf.nn.softplus(-self.labels * self.pred)
        self.l2_loss = tf.reduce_mean(tf.square(self.e1_1)) \
            + tf.reduce_mean(tf.square(self.e1_2)) \
            + tf.reduce_mean(tf.square(self.e2_1)) \
            + tf.reduce_mean(tf.square(self.e2_2)) \
            + tf.reduce_mean(tf.square(self.r_1)) \
            + tf.reduce_mean(tf.square(self.r_2))
        self.loss = tf.add(tf.reduce_mean(losses), self.l2_reg_lambda * self.l2_loss, name="loss")

class Complex_tanh(Complex):
    def add_prediction_op(self):
        self.e1_1 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding1, self.heads))
        self.e1_2 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding2, self.heads))
        self.e2_1 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding1, self.tails))
        self.e2_2 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding2, self.tails))
        self.r_1 = tf.nn.embedding_lookup(self.relation_embedding1, self.relations)
        self.r_2 = tf.nn.embedding_lookup(self.relation_embedding2, self.relations)

        self.pred = tf.subtract(tf.reduce_sum(self.e1_1 * self.r_1 * self.e2_1, -1) +
                                tf.reduce_sum(self.e1_2 * self.r_1 * self.e2_2, -1) +
                                tf.reduce_sum(self.e1_1 * self.r_2 * self.e2_2, -1),
                                tf.reduce_sum(self.e1_2 * self.r_2 * self.e2_1, -1), name="pred")
