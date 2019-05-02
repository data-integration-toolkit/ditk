# coding: utf-8

import random
import numpy as np


class LSTMCNNCRFeeder(object):
    """Helper for feed train data

    Attributes:
        tokens: Tokens [train_size, max_seq_length]
        chars:  Characters [train_size, max_seq_length, max_word_length]
        labels: labels [train_size]
    """

    def __init__(self, tokens,
                 chars,
                 labels,
                 max_seq_length: int,
                 max_char_length: int,
                 batch_size: int):

        self._tokens = tokens
        self._chars = chars
        self._labels = labels
        self._max_seq_length = max_seq_length
        self._max_char_length = max_char_length
        self._batch_size = batch_size

        self.size = len(tokens)
        self.offset = 0
        self.epoch = 1

    @property
    def step_per_epoch(self):
        return (self.size - 1) // self._batch_size + 1

    def next_epoch(self, shuffle=True):
        self.offset = 0
        self.epoch += 1

        # Shuffle
        if shuffle:
            tmp = list(zip(self._tokens, self._chars, self._labels))
            random.shuffle(tmp)
            self._tokens, self._chars, self._labels = zip(*tmp)

    def feed(self):
        next_offset = min(self.size, self.offset + self._batch_size)
        tokens = self._tokens[self.offset: next_offset]
        chars = self._chars[self.offset: next_offset]
        labels = self._labels[self.offset: next_offset]
        self.offset = next_offset

        '''
        Change tokens to (batch_size, max_seq_length)
        Change chars to (batch_size, max_seq_length, max_char_length)
        Change feats to (indices, values, shape)
        Change labels to (batch_size, max_length)
        '''

        tokens = list(map(lambda x: x + ['<pad>'] * (self._max_seq_length - len(x)), tokens))
        tokens = np.array(tokens, dtype=np.str)

        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = np.pad(chars[i][j], (0, self._max_char_length - len(chars[i][j])), 'constant',
                                     constant_values=0)
            for j in range(self._max_seq_length - len(chars[i])):
                chars[i].append(np.zeros((self._max_char_length,), dtype=np.str))
        chars = np.array(chars, dtype=np.str)

        labels = list(map(lambda x: np.pad(x, (0, self._max_seq_length - len(x)), 'constant', constant_values=0),
                          labels))  # Pad zeors
        labels = np.array(labels, dtype=np.int32)

        return tokens, chars, labels

    def predict(self, tokens, chars):
        """
        :param tokens: [length]
        :param feats: [max_length, feat_size]
        :return: (indices, values, shape), len
        """

        length = len(tokens)

        tokens = np.pad(tokens, (0, self._max_seq_length - tokens.shape[0]), 'constant', constant_values=0)
        tokens = np.expand_dims(tokens, 0)

        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = np.pad(chars[i][j], (0, self._max_char_length - len(chars[i][j])), 'constant',
                                     constant_values=0)
            for j in range(self._max_seq_length - len(chars[i])):
                chars[i].append([0] * self._max_char_length)
        chars = np.array(chars, dtype=np.int32)

        return tokens, chars, length

    def test(self, tokens, chars):
        """
        :param tokens: [batch_size, max_length]
        :param feats: [batch_size, max_length, feat_size]
        :return: (indices, values, shape), len
        """

        tokens = list(map(lambda x: np.pad(x, (0, self._max_seq_length - x.shape[0]), 'constant', constant_values=0),
                          tokens))  # Pad zeors
        tokens = np.array(tokens, dtype=np.int32)

        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = np.pad(chars[i][j], (0, self._max_char_length - len(chars[i][j])), 'constant',
                                     constant_values=0)
            for j in range(self._max_seq_length - len(chars[i])):
                chars[i].append([0] * self._max_char_length)
        chars = np.array(chars, dtype=np.int32)

        return tokens, chars

