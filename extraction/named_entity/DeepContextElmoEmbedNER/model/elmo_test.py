# coding: utf-8

import os
import pickle
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from model.bilm.data import Batcher
from model.bilm.model import BidirectionalLanguageModel
from utils.LSTMCNNCRFeeder import LSTMCNNCRFeeder
from utils.parser import parse_conll2003
from utils.conlleval import evaluate
from utils.checkmate import best_checkpoint

from model.Elmo import ElmoModel



def conll2003():
    if not os.path.isfile('../dev/parsedDataDump.pkl'):
        parse_conll2003()
    with open('../dev/parsedDataDump.pkl', 'rb') as fp:
        train_set, val_set, test_set, dicts = pickle.load(fp)

    return train_set, val_set, test_set, dicts


train_set, val_set, test_set, dicts = conll2003()

w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, train_chars, train_la = train_set
val_x, val_chars, val_la = val_set
test_x, test_chars, test_la = test_set

print('Load elmo...')
elmo_batcher = Batcher('../dev/vocab.txt', 50)
elmo_bilm = BidirectionalLanguageModel('../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                                       '../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

print('Load model...')

num_classes = len(la2idx.keys())
max_seq_length = max(
    max(map(len, train_x)),
    max(map(len, test_x)),
)
max_word_length = max(
    max([len(ssc) for sc in train_chars for ssc in sc]),
    max([len(ssc) for sc in test_chars for ssc in sc])
)

model = ElmoModel(
    True,
    50,  # Word embedding size
    16,  # Character embedding size
    200,  # LSTM state size
    128,  # Filter num
    3,  # Filter size
    num_classes,
    max_seq_length,
    max_word_length,
    0.015,
    0.5,
    elmo_bilm,
    1,  # elmo_mode
    elmo_batcher)

print('Start training...')
print('Train size = %d' % len(train_x))
print('Val size = %d' % len(val_x))
print('Test size = %d' % len(test_x))
print('Num classes = %d' % num_classes)

start_epoch = 1
max_epoch = 100

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

best_checkpoint = best_checkpoint('../results/checkpoints/best/', True)
sess.run(tf.tables_initializer())
saver.restore(sess, best_checkpoint)

train_feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, 16)
val_feeder = LSTMCNNCRFeeder(val_x, val_chars, val_la, max_seq_length, max_word_length, 16)
test_feeder = LSTMCNNCRFeeder(test_x, test_chars, test_la, max_seq_length, max_word_length, 16)

preds = []
for step in tqdm(range(val_feeder.step_per_epoch)):
    tokens, chars, labels = val_feeder.feed()
    pred = model.test(sess, tokens, chars)
    preds.extend(pred)
true_seqs = [idx2la[la] for sl in val_la for la in sl]
pred_seqs = [idx2la[la] for sl in preds for la in sl]
ll = min(len(true_seqs), len(pred_seqs))
_, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

val_feeder.next_epoch(False)

print("\nval_f1: %f" % f1)

preds = []
for step in tqdm(range(test_feeder.step_per_epoch)):
    tokens, chars, labels = test_feeder.feed()
    pred = model.test(sess, tokens, chars)
    preds.extend(pred)
true_seqs = [idx2la[la] for sl in test_la for la in sl]
pred_seqs = [idx2la[la] for sl in preds for la in sl]
ll = min(len(true_seqs), len(pred_seqs))
_, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

test_feeder.next_epoch(False)

print("\ntest_f1: %f" % f1)


def dump_topK(prefix, feeder, topK):
    with open('../dev/predict.%s' % prefix, 'w') as fp:
        for _ in tqdm(range(feeder.step_per_epoch)):
            tokens, chars, labels = feeder.feed()

            out = model.decode(sess, tokens, chars, topK)
            for i, preds in enumerate(out):
                length = len(preds[0])

                st = tokens[i, :length].tolist()
                sl = [idx2la[la]+"--" for la in labels[i, :length].tolist()]

                preds = [[idx2la[la] for la in pred] for pred in preds]

                for all in zip(*[st, sl, *preds]):
                    fp.write(' '.join(all) + '\n')
                fp.write('\n')


# dump_topK('train', train_feeder, 2)
# dump_topK('dev', val_feeder, 10)
dump_topK('test', test_feeder, 10)

