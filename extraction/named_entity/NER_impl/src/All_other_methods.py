import functools
import json
import logging
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from collections import Counter
from named_entity.NER_impl.src.tf_metrics import precision, recall, f1


def parse_fn_pred(line):
    # Encode in Bytes for TF
    words = [w.encode() for w in line.strip().split()]

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    return {'words': [words], 'nwords': [len(words)],
            'chars': [chars], 'nchars': [lengths]}

def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),               # (words, nwords)
               ([None, None], [None])),    # (chars, nchars)
              [None])                      # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    (words, nwords), (chars, nchars) = features
    dropout = params['dropout']
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars_embeddings', [num_chars, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                        training=training)

    # Char LSTM
    dim_words = tf.shape(char_embeddings)[1]
    dim_chars = tf.shape(char_embeddings)[2]
    flat = tf.reshape(char_embeddings, [-1, dim_chars, params['dim_chars']])
    t = tf.transpose(flat, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    _, (_, output_fw) = lstm_cell_fw(t, dtype=tf.float32,
                                     sequence_length=tf.reshape(nchars, [-1]))
    _, (_, output_bw) = lstm_cell_bw(t, dtype=tf.float32,
                                     sequence_length=tf.reshape(nchars, [-1]))
    output = tf.concat([output_fw, output_bw], axis=-1)
    char_embeddings = tf.reshape(output, [-1, dim_words, 50])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)


        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    chars = tf.placeholder(dtype=tf.string, shape=[None, None, None],
                           name='chars')
    nchars = tf.placeholder(dtype=tf.int32, shape=[None, None],
                            name='nchars')
    receiver_tensors = {'words': words, 'nwords': nwords,
                        'chars': chars, 'nchars': nchars}
    features = {'words': words, 'nwords': nwords,
                'chars': chars, 'nchars': nchars}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def export_model():
    PARAMS = './results/params.json'
    MODELDIR = './results/model'
    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = 'vocab.words.txt'
    params['chars'] = 'vocab.chars.txt'
    params['tags'] = 'vocab.tags.txt'
    params['glove'] = 'glove.npz'

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    estimator.export_saved_model('saved_model', serving_input_receiver_fn)


def read_dataset_withParse(self, file_dict):
    """
    for file in filename:
        words.txt,tags.txt=utility(file)
    build_vocab()// builds the vocabulary txt file and saves it in local as it maybe required later on
                    to provide output for simple sentences from a pre trained model
    build_glove() // builds glove.npz and saves it in local as it maybe required later on to
                    provide output for simple sentences
    :param filenames:
    :return: data- two files words.txt,tags.txt

    """
    standard_split = ["train", "dev", "test"]
    for split in standard_split:
        fw = open(split + "_words.txt", 'w', encoding="utf-8")
        ft = open(split + "_tags.txt", 'w', encoding="utf-8")
        data = file_dict[split]
        words = ""
        tags = ""
        count = 0
        for line in data:
            if line.strip() == "":
                count += 1
                if count == 1:
                    fw.write(words.rstrip())
                    ft.write(tags.rstrip())
                else:
                    fw.write("\n" + words.rstrip())
                    ft.write("\n" + tags.rstrip())
                words = ""
                tags = ""
            else:
                components = line.split(" ")
                words = words + components[0] + " "
                tags = tags + components[3] + " "
        fw.write("\n"+words.rstrip())
        ft.write("\n"+tags.rstrip())
        fw.close()
        ft.close()
    buildVocab()
    buildGlove()


import numpy as np
MINCOUNT = 1


def words(name):
    return '{}_words.txt'.format(name)
def tags(name):
    return '{}_tags.txt'.format(name)

def buildVocab():
    # print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'dev','test']:
        with Path(words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path('vocab.words.txt').open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    # print('- done. Kept {} out of {}'.format(
    #     len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    # print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path('vocab.chars.txt').open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    vocab_tags = set()
    with Path(tags('train')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path('vocab.tags.txt').open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))

def buildGlove():
    with Path('vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    # glove.840b.300d.txt is downloaded from https://nlp.stanford.edu/projects/glove/
    with Path('glove.840B.300d.txt').open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            # if line_idx % 100000 == 0:
            #     print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    # print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed('glove.npz', embeddings=embeddings)