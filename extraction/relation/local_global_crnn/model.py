#!/usr/bin/env python3
'''
Module Written By Haozhe Gu(Ash)
Model: local_global_CRNN_bio
For the paper: Learning local and global contexts using a convolutional recurrent network model for relation classification in biomedical text
'''


# -*- coding: utf-8 -*-
import relation_extraction_3 as parent
import tensorflow as tf
import numpy as np
import os
import datetime
import time
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import warnings
import pandas as pd
import nltk
import re
import sklearn.exceptions
import random
import uuid
warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.UndefinedMetricWarning)


# Utility Function
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9()<>/,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def myOpen(path):
    print("Reading from :" + path)
    return open(path, 'r')


def myWrite(path):
    print("Writing to: " + path)
    return open(path, 'w')


def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in myOpen(path)]
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        # sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        # sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")
        sentence = sentence.replace(
            "<e1>", "<e1> ").replace("</e1>", " </e11>")
        sentence = sentence.replace(
            "<e2>", "<e2> ").replace("</e2>", " </e22>")

        # tokens = nltk.word_tokenize(sentence)
        #
        # tokens.remove('_/e1_')
        # tokens.remove('_/e2_')
        #
        # e1 = tokens.index("_e1_")
        # del tokens[e1]
        # e2 = tokens.index("_e2_")
        # del tokens[e2]
        #
        # sentence = " ".join(tokens)

        sentence = clean_str(sentence)

        # data.append([id, sentence, e1, e2, relation])
        data.append([id, sentence, relation])

    # df = pd.DataFrame(data=data, columns=["id", "sentence", "e1_pos", "e2_pos", "relation"])
    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    labelsMapping = {'Other': 0,
                     'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                     'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                     'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                     'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                     'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                     'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                     'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                     'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                     'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
    df['label'] = [labelsMapping[r] for r in df['relation']]

    x_text = df['sentence'].tolist()

    # pos1, pos2 = get_relative_position(df)

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()

    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    # return x_text, pos1, pos2, labels
    return x_text, labels


def get_relative_position(df, max_sentence_length=100):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        tokens = nltk.word_tokenize(sentence)
        e1 = df.iloc[df_idx]['e1_pos']
        e2 = df.iloc[df_idx]['e2_pos']

        d1 = ""
        d2 = ""
        for word_idx in range(len(tokens)):
            d1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            d2 += str((max_sentence_length - 1) + word_idx - e2) + " "
        for _ in range(max_sentence_length - len(tokens)):
            d1 += "999 "
            d2 += "999 "
        pos1.append(d1)
        pos2.append(d2)

    return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        # if shuffle:
        # 	shuffle_indices = np.random.permutation(np.arange(data_size))
        # 	shuffled_data = data[shuffle_indices]
        # else:
        # 	shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # yield shuffled_data[start_index:end_index]
            yield data[start_index:end_index]

# Model


class CRNN():
    def __init__(self, layers, max_length, n_classes, pooling_type, vocab_size, embedding_size, f1, f2, n_channels):

        self.input_text = tf.placeholder(
            tf.int32, shape=[None, max_length], name="input_text")
        self.labels = tf.placeholder(tf.int32, shape=[None, n_classes])
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)
        self.pooling_type = pooling_type

        self.W_emb = tf.Variable(
            tf.random_normal([vocab_size, embedding_size]))
        self.text_embedded = tf.nn.embedding_lookup(
            self.W_emb, self.input_text)

        self.length = self.get_length(self.text_embedded)

        layers = list(map(int, layers.split('-')))
        rnn_cell = tf.nn.rnn_cell.LSTMCell
        cells = [rnn_cell(h, activation=tf.tanh, state_is_tuple=True)
                 for h in layers]
        multi_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self.rnn_outputs, _states = tf.nn.bidirectional_dynamic_rnn(
            multi_cells, multi_cells, self.text_embedded, sequence_length=self.length, dtype=tf.float32)
        self.rnn_outputs = tf.concat(self.rnn_outputs, 2)
        self.rnn_outputs = tf.expand_dims(self.rnn_outputs, -1)
        # (64, 100, 200, 1)
        self.first_pooling = tf.nn.max_pool(
            self.rnn_outputs, ksize=[1, f1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        # [batch, in_height, in_width, in_channels] == 64 99 200 1

        # [filter_height, filter_width, in_channels, out_channels]
        W_conv = tf.Variable(tf.truncated_normal(
            [f2, layers[0] * 2, 1, n_channels]))
        b_conv = tf.Variable(tf.truncated_normal([n_channels]))
        self.conv = tf.nn.conv2d(self.first_pooling, W_conv, strides=[
                                 1, 1, 1, 1], padding='VALID')
        self.conv = tf.nn.relu(tf.nn.bias_add(self.conv, b_conv))
        # (64, 95, 1, 100)

        if self.pooling_type == 'max':
            self.max_pooing = tf.nn.max_pool(self.conv, ksize=[
                                             1, max_length - f1 - f2 + 2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            self.max_pooing = tf.nn.dropout(
                self.max_pooing, keep_prob=self.dropout_keep_prob)
            self.pooling = tf.squeeze(self.max_pooing, axis=[1, 2])
        elif self.pooling_type == 'att':
            self.reduced_conv = tf.squeeze(self.conv, axis=2)  # (64, 95, 100)
            W_att = tf.Variable(tf.truncated_normal([n_channels, n_channels]))
            V_att = tf.Variable(tf.truncated_normal([n_channels]))
            self.M_att = tf.tanh(
                tf.einsum('aij,jk->aik', self.reduced_conv, W_att))  # (64, 95, 100)
            self.att_vec = tf.nn.softmax(
                tf.einsum('aij,j->ai', self.M_att, V_att))   # (64, 95)
            self.pooling = tf.einsum(
                'aij,ai->aj', self.reduced_conv, self.att_vec)
            self.pooling = tf.nn.dropout(
                self.pooling, keep_prob=self.dropout_keep_prob)

        self.logits = tf.layers.dense(self.pooling, units=n_classes)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer()
        self.train = self.optimizer.minimize(self.cost)

        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), tf.float32))

    @staticmethod
    def get_length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

# Main class


class local_global_CRNN_bio(parent.RelationExtractionModel):

    def __init__(self, **kwargs):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Parameters
        # ==================================================
        print('=' * 30 + " __INIT__ " + '=' * 30)

        # Data loading params
        tf.app.flags.DEFINE_string(
            "train_test_dir", "/Data/Current_Dataset", "Path of data Folder")
        tf.app.flags.DEFINE_float("test_sample_percentage", .1,
                                  "Percentage of the data to use for testing")
        tf.app.flags.DEFINE_integer("max_sentence_length", 200,
                                    "Max sentence length in train(98)/test(70) data (Default: 100)")

        # Model Hyperparameters
        tf.app.flags.DEFINE_boolean(
            "use_pre_train_embedding", True, "use google pre trained embedding or not")
        tf.app.flags.DEFINE_string("word2vec", "/Data/GoogleNews-vectors-negative300.bin",
                                   "Word2vec file with pre-trained embeddings")
        tf.app.flags.DEFINE_integer("text_embedding_dim", 300,
                                    "Dimensionality of character embedding (Default: 300)")
        # tf.app.flags.DEFINE_integer("position_embedding_dim", 200, "Dimensionality of position embedding (Default: 100)")
        # tf.app.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (Default: 2,3,4,5)")
        # tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (Default: 128)")
        tf.app.flags.DEFINE_string(
            "num_rels", "50", "Number of relation")
        tf.app.flags.DEFINE_string(
            "layers", "100", "Size of rnn output, no (Default: 100")
        tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                                  "Dropout keep probability (Default: 0.5)")
        tf.app.flags.DEFINE_string("pooling_type", "max",
                                   "pooling method, max or att (Default: max)")
        tf.app.flags.DEFINE_float("l2_reg_lambda", 3.0,
                                  "L2 regularization lambda (Default: 3.0)")
        tf.app.flags.DEFINE_integer("f1", 2, "f1 filter size (Default : 2)")
        tf.app.flags.DEFINE_integer("f2", 5, "f2 filter size (Default : 5)")
        tf.app.flags.DEFINE_integer(
            "n_channels", 100, "the number of channels-output vector size, nc(Default : 100")

        # Training parameters
        tf.app.flags.DEFINE_integer(
            "batch_size", 64, "Batch Size (Default: 64)")
        # 100 epochs - 11290 steps
        tf.app.flags.DEFINE_integer(
            "num_epochs", 1, "Number of training epochs (Default: 100)")
        tf.app.flags.DEFINE_integer("display_every", 10,
                                    "Number of iterations to display training info.")
        tf.app.flags.DEFINE_integer("evaluate_every", 100,
                                    "Evaluate model on dev set after this many steps")
        tf.app.flags.DEFINE_integer("checkpoint_every", 100,
                                    "Save model after this many steps")
        tf.app.flags.DEFINE_integer("num_checkpoints", 2,
                                    "Number of checkpoints to store")
        tf.app.flags.DEFINE_float("learning_rate", 1e-3,
                                  "Which learning rate to start with. (Default: 1e-3)")

        # Misc Parameters
        tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                                    "Allow device soft device placement")
        tf.app.flags.DEFINE_boolean("log_device_placement", False,
                                    "Log placement of ops on devices")

        self.FLAGS = tf.app.flags.FLAGS
        self.FLAGS._parse_flags()

        self.FLAGS.train_test_dir = os.path.abspath(os.path.dirname(
            __file__) + self.FLAGS.train_test_dir)
        self.FLAGS.word2vec = os.path.abspath(os.path.dirname(
            __file__) + self.FLAGS.word2vec)
        self.input_data_path = kwargs['inputPath']
        print('-' * 30 + ' Configuration ' + '-' * 30)

        for attr, value in sorted(self.FLAGS.__dict__['__flags'].items()):
            print("{} = {}".format(attr.upper(), value))
        print("")

        print('-' * 30 + " Data/Output " + '-' * 30)
        print('Input Dataset Path: ' + self.input_data_path)
        print('Dataset Folder: ' + self.FLAGS.train_test_dir)
        print('-' * 70)
        if not os.path.exists(self.FLAGS.train_test_dir):
            os.mkdir(self.FLAGS.train_test_dir)

    def get_raw_data(self):
        with myOpen(self.input_data_path) as f:
            self.raw_data = [line.rstrip('\n') for line in f if line.rstrip(
                '\n').split('\t')[-1] != 'None']

    def read_dataset(self):
        print('=' * 30 + " Read_Dataset " + '=' * 30)
        self.get_raw_data()

    def data_preprocess(self):
        print('=' * 30 + " Data_Preprocessing " + '=' * 30)
        random.shuffle(self.raw_data)
        raw_train = self.raw_data[:int((len(self.raw_data) + 1) * 0.90)]
        raw_test = self.raw_data[int(len(self.raw_data) * 0.90 + 1):]
        print("Spliting Dataset to Train: " +
              str(len(raw_train)) + '  Test: ' + str(len(raw_test)))
        with myWrite(self.FLAGS.train_test_dir + '/train.txt') as f:
            for line in raw_train:
                f.write(line + '\n')

        with myWrite(self.FLAGS.train_test_dir + '/test.txt') as f:
            for line in raw_test:
                f.write(line + '\n')

    def get_2id(self):
        trainF = myOpen(self.FLAGS.train_test_dir + '/train.txt')
        testF = myOpen(self.FLAGS.train_test_dir + '/test.txt')
        rels = set()
        ents = set()
        id2ent = {}
        for line in trainF:
            line = line.strip().rstrip('\n')
            it = line.split('\t')
            rels.add(it[-1])
            ents.add(it[1])
            ents.add(it[5])

        for line in testF:
            line = line.strip().rstrip('\n')
            it = line.split('\t')
            rels.add(it[-1])
            ents.add(it[1])
            ents.add(it[5])

        with myWrite(self.FLAGS.train_test_dir + '/relation2id.txt') as f:
            for id, relation in enumerate(rels):
                f.write(relation + '\t' + str(id) + '\n')
        self.FLAGS.num_rels = len(rels)

        with myWrite(self.FLAGS.train_test_dir + '/id2entity.txt') as f:
            for ent in ents:
                id = str(uuid.uuid4())[:8]
                while id in list(id2ent):
                    id = str(uuid.uuid4())[:8]
                f.write(id + '\t' + ent + '\n')

    def get_data(self, isTrain):
        data = []
        if isTrain:
            lines = [line.strip() for line in myOpen(
                self.FLAGS.train_test_dir + '/train.txt')]
        else:
            lines = [line.strip() for line in myOpen(
                self.FLAGS.train_test_dir + '/test.txt')]

        frel = myOpen(self.FLAGS.train_test_dir + '/relation2id.txt')
        # fent = myOpen(train_test_dir + '/id2entity.txt')
        rel2id = {}
        # ent2id = {}

        for line in frel:
            line = line.rstrip('\n')
            it = line.split('\t')
            rel2id[it[0]] = it[1]

        # for line in fent:
        #     line = line.rstrip('\n')
        #     it = line.split('\t')
        #     ent2id[it[1]] = it[0]

        for id, line in enumerate(lines):
            line = line.strip().rstrip('\n')
            it = line.split('\t')
            relation = it[-1]

            sentence = it[0]
            try:
                name1, startPos1, endPos1, name2, startPos2, endPos2 = it[1], int(
                    it[3]), int(it[4]), it[5], int(it[7]), int(it[8])
            except:
                # print("Wrong input format at: " + str(id))
                continue
            sentence = sentence[:startPos1] + '<e1> ' + name1 + ' </e11>' + \
                sentence[endPos1:startPos2] + '<e2> ' + name2 + ' </e22>'

            # data.append([id, sentence, e1, e2, relation])
            sentence = clean_str(sentence)
            data.append([id, sentence, relation])

        df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])

        df['label'] = [int(rel2id[r]) for r in df['relation']]

        x_text = df['sentence'].tolist()

        # pos1, pos2 = get_relative_position(df)

        # Label Data
        y = df['label']
        labels_flat = y.values.ravel()

        labels_count = np.unique(labels_flat).shape[0]

        # convert class labels from scalars to one-hot vectors
        # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
        # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
        # ...
        # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
        def dense_to_one_hot(labels_dense, num_classes):
            num_labels = labels_dense.shape[0]
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

        labels = dense_to_one_hot(labels_flat, labels_count)
        labels = labels.astype(np.uint8)

        # return x_text, pos1, pos2, labels
        return x_text, labels

    def tokenize(self):
        print('=' * 30 + " Tokenization " + '=' * 30)
        self.get_2id()
        with tf.device('/cpu:0'):
            # x_text, pos1, pos2, y = data_helpers.load_data_and_labels(FLAGS.train_dir)
            x_text, y = self.get_data(isTrain=True)
            self.x_test_text, self.y_test = self.get_data(isTrain=False)

            # Build vocabulary
            # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
            # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
            # =>
            # [27 39 40 41 42  1 43  0  0 ... 0]
            # dimension = FLAGS.max_sentence_length
            self.text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
                self.FLAGS.max_sentence_length)
            text_vec = np.array(
                list(self.text_vocab_processor.fit_transform(x_text)))
            # print("Text Vocabulary Size: {:d}".format(
            #     len(self.text_vocab_processor.vocabulary_)))

            # text_vec_test = np.array(
            # list(self.text_vocab_processor.fit_transform(x_test_text)))
            # print("Text Vocabulary Size: {:d}".format(
            #     len(self.text_vocab_processor.vocabulary_)))

            # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
            # [95 96 97 98 99 100 101 999 999 999 ... 999]
            # =>
            # [11 12 13 14 15  16  21  17  17  17 ...  17]
            # dimension = MAX_SENTENCE_LENGTH
            # pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
            # pos_vocab_processor.fit(pos1 + pos2)
            # pos1_vec = np.array(list(pos_vocab_processor.transform(pos1)))
            # pos2_vec = np.array(list(pos_vocab_processor.transform(pos2)))
            # print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))

            # x = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
            x = np.array([list(i) for i in text_vec])
            self.x_train, self.y_train = x, y
            # self.x_test = np.array([list(i) for i in text_vec_test])
            # print("x = {0}".format(x.shape))
            # print("y = {0}".format(y.shape))

            # Randomly shuffle data
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]

            # Split train/test set
            # TODO: This is very crude, should use cross-validation

            dev_sample_index = -1 * \
                int(0.1 * float(len(y)))
            self.x_train, self.x_dev = x_shuffled[:
                                                  dev_sample_index], x_shuffled[dev_sample_index:]
            # x_dev = np.array(x_dev).transpose((1, 0, 2))
            self.y_train, self.y_dev = y_shuffled[:
                                                  dev_sample_index], y_shuffled[dev_sample_index:]

            print(
                "Train/Dev/Test Size: {:d}/{:d}/{:d}".format(len(self.y_train), len(self.x_dev), len(self.y_test)))

    def train(self):
        print('=' * 30 + " Trainning " + '=' * 30)
        with tf.device('/cpu:0'):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                model = CRNN(layers=self.FLAGS.layers, max_length=self.FLAGS.max_sentence_length,
                             n_classes=self.y_train.shape[1], pooling_type=self.FLAGS.pooling_type,
                             vocab_size=len(
                                 self.text_vocab_processor.vocabulary_),
                             embedding_size=self.FLAGS.text_embedding_dim,
                             f1=self.FLAGS.f1, f2=self.FLAGS.f2,
                             n_channels=self.FLAGS.n_channels)

                # Output directory for models and summaries
                timestamp = time.strftime("%H:%M,%b-%d", time.localtime())
                timestamp = timestamp + ',Model:' + self.FLAGS.pooling_type
                out_dir = os.path.abspath(os.path.dirname(
                    __file__) + "/runs/" + timestamp)
                print("Writing to {}\n".format(out_dir))

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(
                    os.path.join(out_dir, "checkpoints"))
                self.checkpoint_dir = checkpoint_dir
                self.text_vocab_dir = out_dir
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(),
                                       max_to_keep=self.FLAGS.num_checkpoints)

                # Write vocabulary
                self.text_vocab_processor.save(
                    os.path.join(out_dir, "text_vocab"))
                # pos_vocab_processor.save(os.path.join(out_dir, "position_vocab"))

                sess.run(tf.global_variables_initializer())

                # Pre-trained word2vec
                if self.FLAGS.use_pre_train_embedding:
                    # initial matrix with random uniform
                    initW = np.random.uniform(
                        -0.25, 0.25, (len(self.text_vocab_processor.vocabulary_), self.FLAGS.text_embedding_dim))
                    # load any vectors from the word2vec
                    print("Load word2vec file {0}".format(self.FLAGS.word2vec))
                    with open(self.FLAGS.word2vec, "rb") as f:
                        header = f.readline()
                        vocab_size, layer1_size = map(int, header.split())
                        binary_len = np.dtype('float32').itemsize * layer1_size
                        for line in range(vocab_size):
                            word = []
                            while True:
                                ch = f.read(1).decode('latin-1')
                                if ch == ' ':
                                    word = ''.join(word)
                                    break
                                if ch != '\n':
                                    word.append(ch)
                            idx = self.text_vocab_processor.vocabulary_.get(
                                word)
                            if idx != 0:
                                initW[idx] = np.fromstring(
                                    f.read(binary_len), dtype='float32')
                            else:
                                f.read(binary_len)
                    sess.run(model.W_emb.assign(initW))
                    print("Success to load pre-trained word2vec model!\n")

                batches = batch_iter(
                    list(zip(self.x_train, self.y_train)), self.FLAGS.batch_size, self.FLAGS.num_epochs)

                max_f1 = -1

                for step, batch in enumerate(batches):
                    x_batch, y_batch = zip(*batch)

                    feed_dict = {model.input_text: x_batch,
                                 model.dropout_keep_prob: self.FLAGS.dropout_keep_prob, model.labels: y_batch}
                    # max_pooling, convs = sess.run([model.max_pooing, model.conv], feed_dict=feed_dict)
                    _, loss, accuracy = sess.run(
                        [model.train, model.cost, model.accuracy], feed_dict=feed_dict)

                    # Training log display
                    if step % self.FLAGS.display_every == 0:
                        print("step {}:, loss {}, acc {}".format(
                            step, loss, accuracy))

                    # Evaluation
                    if step % self.FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        feed_dict = {
                            model.input_text: self.x_dev,
                            model.labels: self.y_dev,
                            model.dropout_keep_prob: 1.0
                        }
                        loss, accuracy, predictions = sess.run(
                            [model.cost, model.accuracy, model.predictions], feed_dict)

                        f1 = f1_score(np.argmax(self.y_dev, axis=1),
                                      predictions, average="micro")
                        recall = recall_score(np.argmax(self.y_dev, axis=1),
                                              predictions, average="micro")
                        print("step {}:, loss {}, acc {}, recall {}, f1 {}\n".format(
                            step, loss, accuracy, recall, f1))

                        # Model checkpoint
                        if f1 > max_f1 * 0.99:
                            path = saver.save(
                                sess, checkpoint_prefix, global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))
                            max_f1 = f1
        return self.FLAGS.train_test_dir + '/prediction.txt'

    def predict(self):
        print('=' * 30 + " Prediction " + '=' * 30)
        # Parameters
        # ==================================================

        # Data loading params
        # tf.app.flags.DEFINE_string(
        #     "eval_dir", "/Data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", "Path of evaluation data")
        # tf.app.flags.DEFINE_string("output_dir", "/Data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/prediction.txt",
        #                            "Path of prediction for evaluation data")
        # tf.app.flags.DEFINE_string("target_dir", "/Data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2//answer.txt",
        #                            "Path of target(answer) file for evaluation data")
        tf.app.flags.DEFINE_string(
            "data_dir", "/Data/Current_Dataset", "Path of data Folder")
        # Eval Parameters
        tf.app.flags.DEFINE_integer(
            "batch_size_eval", 64, "Batch Size (Default: 64)")
        tf.app.flags.DEFINE_string("text_vocab_dir", "",
                                   "Checkpoint directory from training run")

        # Misc Parameters
        tf.app.flags.DEFINE_boolean("allow_soft_placement_eval", True,
                                    "Allow device soft device placement")
        tf.app.flags.DEFINE_boolean("log_device_placement_eval", False,
                                    "Log placement of ops on devices")

        self.FLAGS = tf.app.flags.FLAGS
        self.FLAGS._parse_flags()
        self.FLAGS.text_vocab_dir = self.text_vocab_dir
        self.FLAGS.data_dir = os.path.abspath(os.path.dirname(
            __file__) + self.FLAGS.data_dir)
        # self.FLAGS.eval_dir = os.path.abspath(os.path.dirname(
        #     __file__) + self.FLAGS.eval_dir)
        # self.FLAGS.output_dir = os.path.abspath(os.path.dirname(
        #     __file__) + self.FLAGS.output_dir)
        # self.FLAGS.target_dir = os.path.abspath(os.path.dirname(
        #     __file__) + self.FLAGS.target_dir)
        print("\nParameters:")
        for attr, value in sorted(self.FLAGS.__dict__['__flags'].items()):
            print("{} = {}".format(attr.upper(), value))

        with tf.device('/cpu:0'):
            x_text, y = self.x_test_text, self.y_test

            # Map data into vocabulary
            text_path = os.path.join(
                self.FLAGS.text_vocab_dir, "text_vocab")
            text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
                text_path)
            text_vec = np.array(list(text_vocab_processor.transform(x_text)))

            # Map data into position
            # position_path = os.path.join(FLAGS.checkpoint_dir, "..", "position_vocab")
            # position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
            # pos1_vec = np.array(list(position_vocab_processor.transform(pos1)))
            # pos2_vec = np.array(list(position_vocab_processor.transform(pos2)))

            x_eval = np.array(text_vec)
            y_eval = np.argmax(y, axis=1)

            checkpoint_file = tf.train.latest_checkpoint(
                self.checkpoint_dir)

            frel = myOpen(self.FLAGS.data_dir + '/relation2id.txt')
            # fent = myOpen(train_test_dir + '/id2entity.txt')
            id2rel = {}
            # ent2id = {}

            for line in frel:
                line = line.rstrip('\n')
                it = line.split('\t')
                id2rel[int(it[1])] = it[0]
            graph = tf.Graph()
            with graph.as_default():
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.333)
                session_conf = tf.ConfigProto(gpu_options=gpu_options)
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    # Load the saved meta graph and restore variables
                    saver = tf.train.import_meta_graph(
                        "{}.meta".format(checkpoint_file))
                    saver.restore(sess, checkpoint_file)

                    # Get the placeholders from the graph by name
                    input_text = graph.get_operation_by_name(
                        "input_text").outputs[0]
                    dropout_keep_prob = graph.get_operation_by_name(
                        "dropout_keep_prob").outputs[0]

                    # Tensors we want to evaluate
                    predictions = graph.get_operation_by_name(
                        "predictions").outputs[0]

                    # Generate batches for one epoch
                    batches = batch_iter(
                        list(x_eval), self.FLAGS.batch_size_eval, 1, shuffle=False)

                    # Collect the predictions here
                    all_predictions = []
                    for x_eval_batch in batches:
                        # x_batch = np.array(x_eval_batch).transpose((1, 0, 2))
                        batch_predictions = sess.run(predictions, {input_text: x_eval_batch,
                                                                   dropout_keep_prob: 1.0})
                        all_predictions = np.concatenate(
                            [all_predictions, batch_predictions])

                    with myWrite(self.FLAGS.data_dir + "/prediction.txt") as f:
                        for idx, prd in enumerate(all_predictions):
                            sentence = x_text[idx]
                            ent1 = re.search('<e1> (.+) </e11>', sentence)
                            ent2 = re.search('<e2> (.+) </e22>', sentence)
                            if ent1 and ent2:
                                ent1 = ent1.group(1)
                                ent2 = ent2.group(1)
                            else:
                                print("Entitity Extraction Fail")
                                continue
                            sentence = re.sub(
                                '<e1> (.+) </e11>', ent1, sentence)
                            sentence = re.sub(
                                '<e2> (.+) </e22>', ent2, sentence)
                            pred_label = id2rel[int(prd)]
                            true_label = id2rel[int(y_eval[idx])]
                            f.write(sentence + '\t' + ent1 + '\t' + ent2 + '\t' + pred_label
                                    + '\t' + true_label + '\n')
                    self.y_prediction, self.all_prediction = y_eval, all_predictions
        return self.FLAGS.data_dir + "/prediction.txt"

    def evaluate(self):
        print('=' * 30 + " Evaluation " + '=' * 30)

        print("\nTotal number of test examples: {}".format(len(self.y_prediction)))
        print("Precsion Score: {:g}".format(
            precision_score(self.y_prediction, self.all_prediction,  average="micro")))
        print("Recall Score: {:g}".format(
            recall_score(self.y_prediction, self.all_prediction,  average="micro")))
        print("F1 Score: {:g}".format(
            f1_score(self.y_prediction, self.all_prediction, average="micro")))
        # correct_predictions = float(sum(all_predictions == y_eval))
        # print("Total number of test examples: {}".format(len(y_eval)))
        # print("Accuracy: {:g}".format(
        #     correct_predictions / float(len(y_eval))))
        # print()
        # print(
        #     "Macro-Average F1 Score: {:g}".format(f1_score(y_eval, all_predictions, average="macro")))

        # labelsMapping = {0: 'Other',
        #                  1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
        #                  3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
        #                  5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
        #                  7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
        #                  9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
        #                  11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
        #                  13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
        #                  15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
        #                  17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}
        # output_file = open(self.FLAGS.output_dir, 'w')
        # target_file = open(self.FLAGS.target_dir, 'w')
        # for i in range(len(all_predictions)):
        #     output_file.write("{}\t{}\n".format(
        #         i, labelsMapping[all_predictions[i]]))
        #     target_file.write("{}\t{}\n".format(
        #         i, labelsMapping[y_eval[i]]))
        # output_file.close()
        # target_file.close()

        # correct_predictions = float(sum(all_predictions == y_eval))
