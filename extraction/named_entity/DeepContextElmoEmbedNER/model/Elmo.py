# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from model.bilm.elmo import weight_layers
from utils.utils import viterbi_decode_topk, decay_learning_rate, \
    load_train_vocab, load_pretrained_glove, load_pretrained_senna


class ElmoModel(object):
    """Bi-LSTM + ElMo + CNN + CRF implemented by Tensorflow

    Attributes:
        num_classes: number of classes
        max_seq_length: max length of sentence
        max_word_length: max length of word
        learning_rate: learning rate
    """

    def __init__(self, pre_embedding: bool,
                 word_embed_size: int,
                 char_embed_size: int,
                 hidden_size: int,
                 filter_num: int,
                 filter_size: int,
                 num_classes: int,
                 max_seq_length: int,
                 max_word_length: int,
                 learning_rate: float,
                 dropout: float,
                 elmo_bilm,
                 elmo_mode,
                 elmo_batcher,
                 **kwargs):
        self.pre_embedding = pre_embedding
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.hidden_size = hidden_size
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
        self.learning_rate = learning_rate
        self._dropout = dropout

        self.elmo_bilm = elmo_bilm
        self.elmo_mode = elmo_mode
        self.elmo_batcher = elmo_batcher

        self._build_graph(**kwargs)

    def _add_placeholders(self):
        self.tokens = tf.placeholder(tf.string, [None, self.max_seq_length])
        self.chars = tf.placeholder(tf.string, [None, self.max_seq_length, self.max_word_length])
        self.dropout = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.int32, [None, self.max_seq_length])

        # elmo
        self.elmo_p = tf.placeholder(tf.int32, [None, None, None])

    def _add_embedding(self, **kwargs):
        with tf.variable_scope('embedding'):
            train_word_vocab, train_char_vocab = load_train_vocab(**kwargs)
            if self.pre_embedding:
                # pretrained_vocab, pretrained_embs = load_pretrained_senna()
                pretrained_vocab, pretrained_embs = load_pretrained_glove(**kwargs)

                only_in_train_vocab = list(set(train_word_vocab) - set(pretrained_vocab))
                vocab = ['<pad>'] + pretrained_vocab + only_in_train_vocab

                vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(vocab),
                    default_value=len(vocab)
                )
                word_string_tensor = vocab_lookup.lookup(self.tokens)

                pad_embs = tf.get_variable(
                    name='embs_pad',
                    shape=[1, self.word_embed_size],
                    initializer=tf.initializers.zeros(),
                    trainable=True
                )

                pretrained_embs = tf.get_variable(
                    name='embs_pretrained',
                    initializer=tf.constant_initializer(np.asarray(pretrained_embs), dtype=tf.float32),
                    shape=pretrained_embs.shape,
                    trainable=True
                )
                train_embs = tf.get_variable(
                    name='embs_only_in_train',
                    shape=[len(only_in_train_vocab), self.word_embed_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=True
                )
                unk_embs = tf.get_variable(
                    name='embs_unk',
                    shape=[1, self.word_embed_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=True
                )
                word_embeddings = tf.concat([pad_embs, pretrained_embs, train_embs, unk_embs], axis=0)
            else:
                word_embeddings = tf.get_variable(
                    name='embeds_word',
                    shape=[len(train_word_vocab) + 1, self.word_embed_size])
                vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(train_word_vocab),
                    default_value=len(train_word_vocab)
                )
                word_string_tensor = vocab_lookup.lookup(self.tokens)

            self.length = tf.count_nonzero(word_string_tensor, axis=1)
            self.word_embedding_layer = tf.nn.embedding_lookup(word_embeddings, word_string_tensor)

            char_embeddings = tf.get_variable(
                name='embs_char',
                shape=[len(train_char_vocab) + 1, self.char_embed_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True
            )
            vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(train_char_vocab),
                default_value=len(train_char_vocab)
            )
            char_string_tensor = vocab_lookup.lookup(self.chars)

            self.char_embedding_layer = tf.nn.embedding_lookup(char_embeddings, char_string_tensor)

    def _add_cnn(self):
        with tf.variable_scope('cnn'):
            cnn_inputs = tf.reshape(self.char_embedding_layer, (-1, self.max_word_length, self.char_embed_size, 1))

            # A dropout layer is applied before character embeddings are input to CNN
            cnn_inputs = tf.nn.dropout(cnn_inputs, keep_prob=self.dropout)

            filter_shape = [self.filter_size, self.char_embed_size, 1, self.filter_num]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="cnn_w")
            b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name="cnn_b")
            conv = tf.nn.conv2d(
                cnn_inputs,
                w,
                strides=[1, 1, 1, 1],
                padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="cnn_relu")
            pool = tf.nn.max_pool(
                h,
                ksize=[1, self.max_word_length - self.filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID')

            pool = tf.reshape(pool, (-1, self.max_seq_length, self.filter_num))

        self.embedding_layer = tf.concat([self.word_embedding_layer, pool], 2)

    def _add_elmo_embedding(self):
        """
        The Elmo embedding layer
        """
        embeddings_op = self.elmo_bilm(self.elmo_p)
        self.elmo_emb = weight_layers('input', embeddings_op, l2_coef=0.0)['weighted_op']

        # self.elmo_emb = tc.layers.fully_connected(self.elmo_emb, num_outputs=self.hidden_size, activation_fn=None)

        # self.elmo_emb = tf.nn.dropout(self.elmo_emb, self.dropout)

        if self.elmo_mode == 1:
            # concat word emb and elmo emb
            self.embedding_layer = tf.concat([self.embedding_layer, self.elmo_emb], 2)
        else:
            # Default: only use Elmo
            self.embedding_layer = self.elmo_emb

    def _add_rnn(self):
        def rnn_cell(hidden_size=self.hidden_size, gru=True):
            if gru:
                cell = tf.contrib.rnn.GRUCell(hidden_size,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
            return cell

        # dropout layers are applied on both the input and output vectors of BiLSTM
        rnn_inputs = tf.nn.dropout(self.embedding_layer, keep_prob=self.dropout)

        with tf.variable_scope('recurrent'):
            fw_cells = [rnn_cell(200), rnn_cell(200)]
            bw_cells = [rnn_cell(200), rnn_cell(200)]
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells,
                rnn_inputs,
                dtype=tf.float32,
                sequence_length=self.length
            )
            self.layer_output = tf.concat(values=outputs, axis=2)

    def _add_crf(self):
        flattened_output = tf.reshape(self.layer_output, [-1, self.hidden_size * 2])
        with tf.variable_scope('linear'):
            w = tf.get_variable('w', [self.hidden_size * 2, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [self.num_classes], initializer=tf.contrib.layers.xavier_initializer())

        flattened_potentials = tf.matmul(flattened_output, w) + b
        self.unary_potentials = tf.reshape(
            flattened_potentials,
            [-1, self.max_seq_length, self.num_classes]
        )
        self.ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(
            self.unary_potentials, self.labels, self.length
        )
        self.viterbi_sequence, _ = tf.contrib.crf.crf_decode(
            self.unary_potentials, self.trans_params, tf.cast(self.length, tf.int32)
        )
        self.loss = -tf.reduce_sum(self.ll)

    def _add_train_op(self):
        # learning_rate = decay_learning_rate(self.learning_rate,
        #                                     self.global_step,
        #                                     878,
        #                                     0.05)
        # self.lr = learning_rate
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(0.001)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
        self._train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params),
            global_step=self.global_step
        )

    def _build_graph(self, **kwargs):
        self.global_step = tf.Variable(0, trainable=False)
        self._add_placeholders()
        self._add_embedding(**kwargs)
        self._add_cnn()

        # elmo
        self._add_elmo_embedding()

        self._add_rnn()
        self._add_crf()
        self._add_train_op()

    def train_step(self, sess, tokens, chars, labels):

        lower_tokens = np.char.lower(tokens)

        input_feed = {
            self.tokens: lower_tokens,
            self.chars: chars,
            self.labels: labels,
            self.dropout: self._dropout
        }

        if self.elmo_bilm and self.elmo_batcher:
            input_feed[self.elmo_p] = self.elmo_batcher.batch_sentences([sx.tolist() for sx in tokens])

        output_feed = [
            self._train_op,
            self.loss
        ]

        _, loss = sess.run(output_feed, input_feed)

        return loss

    def test(self, sess, tokens, chars):

        lower_tokens = np.char.lower(tokens)

        input_feed = {
                self.tokens: lower_tokens,
                self.chars: chars,
                self.dropout: 1.0
            }

        if self.elmo_bilm and self.elmo_batcher:
            input_feed[self.elmo_p] = self.elmo_batcher.batch_sentences([sx.tolist() for sx in tokens])

        viterbi_sequences, lengths = sess.run(
            [self.viterbi_sequence, self.length],
            input_feed
        )

        pred = []
        for i in range(lengths.shape[0]):
            length = lengths[i]
            pred.append(viterbi_sequences[i, :length])

        return pred

    def decode(self, sess, tokens, chars, topK=5):
        """
        score: [seq_len, num_tags]
        transition_params: [num_tags, num_tags]
        """

        lower_tokens = np.char.lower(tokens)

        input_feed = {
            self.tokens: lower_tokens,
            self.chars: chars,
            self.dropout: 1.0
        }

        if self.elmo_bilm and self.elmo_batcher:
            input_feed[self.elmo_p] = self.elmo_batcher.batch_sentences([sx.tolist() for sx in tokens])

        scores, trans_params, lengths = sess.run([self.unary_potentials, self.trans_params, self.length], input_feed)

        out = []
        for score, length in zip(scores, lengths):
            score = score[:length, :]

            viterbi, viterbi_score = viterbi_decode_topk(
                score,
                trans_params,
                topK
            )

            out.append(viterbi)

        '''
        viterbi, viterbi_score = tf.contrib.crf.viterbi_decode(
            score,
            trans_params
        )
        print("{:<20} {}".format(viterbi_score, viterbi))
        '''

        '''
        viterbi, viterbi_score = viterbi_decode_topk(
            score,
            trans_params,
            topK
        )
        '''

        # for a, b in zip(viterbi_score, viterbi):
        #     print("{:<20} {}".format(a, b))

        # return viterbi, viterbi_score

        return out
