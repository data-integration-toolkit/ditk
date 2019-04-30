# coding: utf-8

import re
import pickle
import numpy as np

from utils.utils import load_pretrained_glove, load_pretrained_senna

max_word_len = 30


def parse_conll2003(train_loc='data/train.txt', test_loc='data/test.txt', val_loc='data/valid.txt', dir_root=''):
    train_file = open(dir_root+train_loc)
    val_file = open(dir_root+val_loc)
    test_file = open(dir_root+test_loc)

    word_set = set()
    char_set = set()
    label_set = set()

    vocab = set()

    files = [('train', train_file), ('val', val_file), ('test', test_file)]
    dump = []

    for prefix, file in files:
        x = []
        ch = []
        la = []

        sx = []  # sentence x
        sc = []  # sentence char
        sl = []  # sentence label
        for row in file:
            if row == '\n':
                if sx:
                    x.append(sx)
                    ch.append(sc)
                    la.append(sl)

                    sx = []
                    sc = []
                    sl = []
            else:
                data = row.split(' ')
                token = data[0]
                token = re.sub('\d', '0', token)

                chars = [ch for ch in token]  # Character一定要保留大小写

                # token = token.lower()  # 可选：是否全部转小写

                label = data[-1].strip()

                if len(chars) > max_word_len:
                    # half = max_word_len // 2
                    # chars = chars[:half] + chars[-(max_word_len - half):]
                    chars = chars[:max_word_len]

                sx.append(token)
                sc.append(chars)
                sl.append(label)

                if prefix == 'train':
                    # Should only update word in train set
                    word_set.add(token.lower())  # word embedding词汇表只要小写
                    char_set.update(*chars)
                    label_set.add(label)
                vocab.add(token)  # elmo embedding词汇表大小写都要
        dump.append([x, ch, la])

    w2idx = {}
    ch2idx = {}
    la2idx = {}

    with open(dir_root+'dev/train.word.vocab', 'w') as fp:
        for idx, word in enumerate(sorted(word_set)):
            w2idx[word] = idx
            fp.write(word + '\n')

    with open(dir_root+'dev/vocab.txt', 'w', encoding='gb18030') as fp:
        vocab = sorted(vocab)

        vocab.insert(0, '<S>')
        vocab.insert(1, '</S>')
        vocab.insert(2, '<UNK>')

        for word in vocab:
            fp.write(word + '\n')

    with open(dir_root+'dev/train.char.vocab', 'w') as fp:
        for idx, char in enumerate(sorted(char_set)):
            ch2idx[char] = idx
            fp.write(char + '\n')

    for idx, label in enumerate(sorted(label_set)):
        la2idx[label] = idx

    for i in range(3):
        # dump[i][1] = [[[ch2idx.get(ch, len(ch2idx)) for ch in ssc] for ssc in sc] for sc in dump[i][1]]  # char
        dump[i][2] = [np.array([la2idx[la] for la in sl]) for sl in dump[i][2]]  # label

    with open(dir_root+'dev/conll.pkl', 'wb') as fp:
        pickle.dump((dump[0], dump[1], dump[2],
                     {
                         'words2idx': w2idx,
                         'chars2idx': ch2idx,
                         'labels2idx': la2idx
                     }), fp)


def parse_data(data, dir_root='', data_type='single'):

    if data_type == 'single':
        pass

    elif data_type == 'multiple':
        pass

    train_file = open(dir_root + train_loc)
    val_file = open(dir_root + val_loc)
    test_file = open(dir_root + test_loc)

    word_set = set()
    char_set = set()
    label_set = set()

    vocab = set()

    files = [('train', train_file), ('val', val_file), ('test', test_file)]
    dump = []

    for prefix, file in files:
        x = []
        ch = []
        la = []

        sx = []  # sentence x
        sc = []  # sentence char
        sl = []  # sentence label
        for row in file:
            if row == '\n':
                if sx:
                    x.append(sx)
                    ch.append(sc)
                    la.append(sl)

                    sx = []
                    sc = []
                    sl = []
            else:
                data = row.split(' ')
                token = data[0]
                token = re.sub('\d', '0', token)

                chars = [ch for ch in token]  # Character一定要保留大小写

                # token = token.lower()  # 可选：是否全部转小写

                label = data[-1].strip()

                if len(chars) > max_word_len:
                    # half = max_word_len // 2
                    # chars = chars[:half] + chars[-(max_word_len - half):]
                    chars = chars[:max_word_len]

                sx.append(token)
                sc.append(chars)
                sl.append(label)

                if prefix == 'train':
                    # Should only update word in train set
                    word_set.add(token.lower())  # word embedding词汇表只要小写
                    char_set.update(*chars)
                    label_set.add(label)
                vocab.add(token)  # elmo embedding词汇表大小写都要
        dump.append([x, ch, la])

    w2idx = {}
    ch2idx = {}
    la2idx = {}

    with open(dir_root + 'dev/train.word.vocab', 'w') as fp:
        for idx, word in enumerate(sorted(word_set)):
            w2idx[word] = idx
            fp.write(word + '\n')

    with open(dir_root + 'dev/vocab.txt', 'w', encoding='gb18030') as fp:
        vocab = sorted(vocab)

        vocab.insert(0, '<S>')
        vocab.insert(1, '</S>')
        vocab.insert(2, '<UNK>')

        for word in vocab:
            fp.write(word + '\n')

    with open(dir_root + 'dev/train.char.vocab', 'w') as fp:
        for idx, char in enumerate(sorted(char_set)):
            ch2idx[char] = idx
            fp.write(char + '\n')

    for idx, label in enumerate(sorted(label_set)):
        la2idx[label] = idx

    for i in range(3):
        # dump[i][1] = [[[ch2idx.get(ch, len(ch2idx)) for ch in ssc] for ssc in sc] for sc in dump[i][1]]  # char
        dump[i][2] = [np.array([la2idx[la] for la in sl]) for sl in dump[i][2]]  # label

    with open(dir_root + 'dev/conll.pkl', 'wb') as fp:
        pickle.dump((dump[0], dump[1], dump[2],
                     {
                         'words2idx': w2idx,
                         'chars2idx': ch2idx,
                         'labels2idx': la2idx
                     }), fp)

    pass
