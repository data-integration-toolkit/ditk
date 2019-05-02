import codecs
import json
import os
import re
import sys
import time
from collections import defaultdict
from glob import glob
from os import listdir

import numpy as np
import tensorflow as tf

import extraction.named_entity.DilatedCNN.src.eval_f1 as evaluation
import extraction.named_entity.DilatedCNN.src.tf_utils as tf_utils
from extraction.named_entity.ner import Ner
from extraction.named_entity.DilatedCNN.src.bilstm import BiLSTM as BiLSTM
from extraction.named_entity.DilatedCNN.src.bilstm_char import BiLSTMChar as BiLSTMChar
from extraction.named_entity.DilatedCNN.src.cnn import CNN as CNN
from extraction.named_entity.DilatedCNN.src.cnn_char import CNNChar as CNNChar
from extraction.named_entity.DilatedCNN.src.data_utils import Batcher as Batcher, SeqBatcher as SeqBatcher


class DilatedCNN(Ner):
    file_list_lample = []
    file_list_in = []
    embedding = ""
    vocabs = ''
    model_path = None
    tag_map = defaultdict(lambda: 'O')

    def del_all_flags(self):
        FLAGS = tf.app.flags.FLAGS
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    def convert_ground_truth(self, data, *args, **kwargs):
        """
        this method is for converting words into tags. This method should be invoked after using read dataset method
        :param data: a list of words 
        :param args: 
        :param kwargs: 
        :return: a list of tags 
        """
        result = []
        with open(data, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    result.append(parts[-1])
        return result

    def read_dataset(self, input_files=None, embedding='./data/embeddings/glove.6B.100d.txt', *args, **kwargs):
        """
        This method is for reading the data set of training, validating and testing
        :param input_files: an array containing all relative files, trainging, validating, and testing 
        :param embedding: the embedding file path
        for example:
        /Users/liyiran/ditk/extraction/named_entity/yiran/embedding/glove.6B.100d.txt
        :param args: 
        :param kwargs: 
        :return: 
        :raise if the length of input files is not 3 or files do not exist
        """
        self.embedding = embedding
        if input_files is None or not len(input_files) is 3:
            raise NameError("input files should contain 3 files for training, valid and test ")
        self.file_list_in.append(input_files[0])
        self.file_list_in.append(input_files[1])
        self.file_list_in.append(input_files[2])
        for i in range(0, 3):
            with open(input_files[i], 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        self.tag_map[parts[0]] = parts[-1]
        path = '/'.join(input_files[0].split('/')[:-1])
        self.model_path = path + '/models/dilated-cnn'
        lample_test = path + '/conll2003-w3-lample/test.txt'
        lample_valid = path + '/conll2003-w3-lample/valid.txt'
        lample_train = path + '/conll2003-w3-lample/train.txt'
        # self.file_list_in.append(train_dir + '/conll2003/test.txt')
        # self.file_list_in.append(train_dir + '/conll2003/valid.txt')
        # self.file_list_in.append(train_dir + '/conll2003/train.txt')
        self.file_list_lample.append(lample_train)
        self.file_list_lample.append(lample_valid)
        self.file_list_lample.append(lample_test)
        vocabs = path + '/vocabs'
        if not os.path.exists(lample_test):
            os.makedirs(lample_test)
        if not os.path.exists(lample_valid):
            os.makedirs(lample_valid)
        if not os.path.exists(lample_train):
            os.makedirs(lample_train)
        if not os.path.exists(vocabs):
            os.makedirs(vocabs)
        cut_off = defaultdict(lambda: 0)
        with codecs.open(input_files[0], 'r', 'utf8') as reader:
            for line in reader:
                line = line.strip()
                if line:
                    parts = line.split()
                    word = parts[0]
                    word = re.sub('[0-9]', '0', word)
                    cut_off[word] += 1
        cut_off = {k for k, v in cut_off.items() if v >= 4}
        print(len(cut_off))
        with open(path + '/vocabs/conll2003_cutoff_4.txt', 'w+') as f:
            self.vocabs = path + '/vocabs/conll2003_cutoff_4.txt'
            for item in cut_off:
                f.write("%s\n" % item)

    def train(self, data, *args, **kwargs):
        """
        This method is for training the model and the model will be stored in the model folder
        :param data: it is not used since the traning data set will be stored when read dataset is invoked 
        :param args: 
        :param kwargs: 
        :return: 
        """

        def shape(string):
            if all(c.isupper() for c in string):
                return "AA"
            if string[0].isupper():
                return "Aa"
            if any(c for c in string if c.isupper()):
                return "aAa"
            else:
                return "a"

        def get_str_label_from_line_conll(line):
            token_str, _, _, label_str = line.strip().split(' ')
            return token_str, label_str, ''

        def get_str_label_from_line_ontonotes(line, current_tag):
            parts = line.split()
            try:
                token_str = parts[3]
                onto_label_str = parts[10]
                if onto_label_str.startswith('('):
                    if onto_label_str.endswith(')'):
                        label_str = 'U-%s' % onto_label_str[1:-1]
                        current_tag = ''
                    else:
                        current_tag = onto_label_str[1:-1]
                        label_str = 'B-%s' % current_tag
                elif onto_label_str.endswith(')'):
                    label_str = 'L-%s' % current_tag
                    current_tag = ''
                elif current_tag != '':
                    label_str = 'I-%s' % current_tag
                else:
                    label_str = 'O'
                return token_str, label_str, current_tag
            except Exception as e:
                print("Caught exception: %s" % e.message)
                print("Line: %s" % line)
                raise

        def make_example(writer, lines, label_map, token_map, shape_map, char_map, update_vocab, update_chars):
            # data format is:
            # token pos phrase ner
            # LONDON NNP I-NP I-LOC
            # 1996-08-30 CD I-NP O

            # West NNP I-NP I-MISC
            # Indian NNP I-NP I-MISC
            # all-rounder NN I-NP O
            # ...

            sent_len = len(lines)
            num_breaks = sum([1 if line.strip() == "" else 0 for line in lines])
            max_len_with_pad = pad_width * (num_breaks + 2) + (sent_len - num_breaks)
            max_word_len = max(map(len, lines))

            oov_count = 0
            if sent_len == 0:
                return 0, 0, 0

            tokens = np.zeros(max_len_with_pad, dtype=np.int64)
            shapes = np.zeros(max_len_with_pad, dtype=np.int64)
            chars = np.zeros(max_len_with_pad * max_word_len, dtype=np.int64)
            intmapped_labels = np.zeros(max_len_with_pad, dtype=np.int64)
            sent_lens = []
            tok_lens = []

            # initial padding
            tokens[:pad_width] = token_map[PAD_STR]
            shapes[:pad_width] = shape_map[PAD_STR]
            chars[:pad_width] = char_map[PAD_STR]
            if FLAGS.predict_pad:
                intmapped_labels[:pad_width] = label_map[PAD_STR]
            tok_lens.extend([1] * pad_width)

            last_label = "O"
            labels = []
            current_sent_len = 0
            char_start = pad_width
            idx = pad_width
            current_tag = ''
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    token_str, label_str, current_tag = get_str_label_from_line_conll(line) if FLAGS.dataset == 'conll2003' else get_str_label_from_line_ontonotes(line, current_tag)

                    # skip docstart markers
                    if token_str == DOC_MARKER:
                        return 0, 0, 0

                    current_sent_len += 1

                    # process tokens to match Collobert embedding preprocessing:
                    # - normalize the digits to 0
                    # - lowercase
                    token_str_digits = re.sub("\d", "0", token_str)

                    # get capitalization features
                    token_shape = shape(token_str_digits)

                    token_str_normalized = token_str_digits.lower() if FLAGS.lowercase else token_str_digits

                    if token_shape not in shape_map:
                        shape_map[token_shape] = len(shape_map)

                    # Don't use normalized token str -- want digits
                    for char in token_str:
                        if char not in char_map and update_chars:
                            char_map[char] = len(char_map)
                            char_int_str_map[char_map[char]] = char
                    tok_lens.append(len(token_str))

                    # convert label to BILOU encoding
                    label_bilou = label_str
                    # handle cases where we need to update the last token we processed
                    if label_str == "O" or label_str[0] == "B" or (last_label != "O" and label_str[2] != last_label[2]):
                        if last_label[0] == "I":
                            labels[-1] = "L" + labels[-1][1:]
                        elif last_label[0] == "B":
                            labels[-1] = "U" + labels[-1][1:]
                    if label_str[0] == "I":
                        if last_label == "O" or label_str[2] != last_label[2]:
                            label_bilou = "B-" + label_str[2:]

                    if token_str_normalized not in token_map:
                        oov_count += 1
                        if update_vocab:
                            token_map[token_str_normalized] = len(token_map)
                            token_int_str_map[token_map[token_str_normalized]] = token_str_normalized

                    tokens[idx] = token_map.get(token_str_normalized, token_map[OOV_STR])
                    shapes[idx] = shape_map[token_shape]
                    chars[char_start:char_start + tok_lens[-1]] = [char_map.get(char, char_map[OOV_STR]) for char in token_str]
                    char_start += tok_lens[-1]
                    labels.append(label_bilou)
                    last_label = label_bilou
                    idx += 1
                elif current_sent_len > 0:
                    sent_lens.append(current_sent_len)
                    current_sent_len = 0
                    tokens[idx:idx + pad_width] = token_map[PAD_STR]
                    shapes[idx:idx + pad_width] = shape_map[PAD_STR]
                    chars[char_start:char_start + pad_width] = char_map[PAD_STR]
                    char_start += pad_width
                    tok_lens.extend([1] * pad_width)
                    labels.extend([PAD_STR if FLAGS.predict_pad else "O"] * pad_width)
                    idx += pad_width

                    last_label = "O"

            if last_label[0] == "I":
                labels[-1] = "L" + labels[-1][1:]
            elif last_label[0] == "B":
                labels[-1] = "U" + labels[-1][1:]

            if not FLAGS.documents:
                sent_lens.append(sent_len)

            # final padding
            if not FLAGS.documents:
                tokens[idx:idx + pad_width] = token_map[PAD_STR]
                shapes[idx:idx + pad_width] = shape_map[PAD_STR]
                chars[char_start:char_start + pad_width] = char_map[PAD_STR]
                char_start += pad_width
                tok_lens.extend([1] * pad_width)
                if FLAGS.predict_pad:
                    intmapped_labels[idx:idx + pad_width] = label_map[PAD_STR]

            for label in labels:
                if label not in label_map:
                    label_map[label] = len(label_map)
                    label_int_str_map[label_map[label]] = label

            # intmapped_labels[pad_width:pad_width+len(labels)] = map(lambda s: label_map[s], labels)
            intmapped_labels[pad_width:pad_width + len(labels)] = [label_map[s] for s in labels]

            # chars = chars.flatten()

            # print(sent_lens)

            padded_len = (len(sent_lens) + 1) * pad_width + sum(sent_lens)
            intmapped_labels = intmapped_labels[:padded_len]
            tokens = tokens[:padded_len]
            shapes = shapes[:padded_len]
            chars = chars[:sum(tok_lens)]

            if FLAGS.debug:
                print("sent lens: ", sent_lens)
                print("tok lens: ", tok_lens, len(tok_lens), sum(tok_lens))
                print("labels", map(lambda t: label_int_str_map[t], intmapped_labels), len(intmapped_labels))
                print("tokens", map(lambda t: token_int_str_map[t], tokens), len(tokens))
                print("chars", map(lambda t: char_int_str_map[t], chars), len(chars))

            example = tf.train.SequenceExample()

            fl_labels = example.feature_lists.feature_list["labels"]
            for l in intmapped_labels:
                fl_labels.feature.add().int64_list.value.append(l)

            fl_tokens = example.feature_lists.feature_list["tokens"]
            for t in tokens:
                fl_tokens.feature.add().int64_list.value.append(t)

            fl_shapes = example.feature_lists.feature_list["shapes"]
            for s in shapes:
                fl_shapes.feature.add().int64_list.value.append(s)

            fl_chars = example.feature_lists.feature_list["chars"]
            for c in chars:
                fl_chars.feature.add().int64_list.value.append(c)

            fl_seq_len = example.feature_lists.feature_list["seq_len"]
            for seq_len in sent_lens:
                fl_seq_len.feature.add().int64_list.value.append(seq_len)

            fl_tok_len = example.feature_lists.feature_list["tok_len"]
            for tok_len in tok_lens:
                fl_tok_len.feature.add().int64_list.value.append(tok_len)

            writer.write(example.SerializeToString())
            return sum(sent_lens), oov_count, 1

        def tsv_to_examples():
            label_map = {}
            token_map = {}
            shape_map = {}
            char_map = {}

            update_vocab = True
            update_chars = True

            token_map[PAD_STR] = len(token_map)
            token_int_str_map[token_map[PAD_STR]] = PAD_STR
            char_map[PAD_STR] = len(char_map)
            char_int_str_map[char_map[PAD_STR]] = PAD_STR
            shape_map[PAD_STR] = len(shape_map)
            if FLAGS.predict_pad:
                label_map[PAD_STR] = len(label_map)
                label_int_str_map[label_map[PAD_STR]] = PAD_STR

            token_map[OOV_STR] = len(token_map)
            token_int_str_map[token_map[OOV_STR]] = OOV_STR
            char_map[OOV_STR] = len(char_map)
            char_int_str_map[char_map[OOV_STR]] = OOV_STR

            # load vocab if we have one
            if FLAGS.vocab != '':
                update_vocab = False
                with open(FLAGS.vocab, 'r') as f:
                    for line in f.readlines():
                        word = line.strip().split(" ")[0]
                        if word not in token_map:
                            # print("adding word %s" % word)
                            token_map[word] = len(token_map)
                            token_int_str_map[token_map[word]] = word
            if FLAGS.update_vocab != '':
                with open(FLAGS.update_vocab, 'r') as f:
                    for line in f.readlines():
                        word = line.strip().split(" ")[0]
                        if word not in token_map:
                            # print("adding word %s" % word)
                            token_map[word] = len(token_map)
                            token_int_str_map[token_map[word]] = word

            # load labels if given
            if FLAGS.labels != '':
                with open(FLAGS.labels, 'r') as f:
                    for line in f.readlines():
                        label, idx = line.strip().split("\t")
                        label_map[label] = int(idx)
                        label_int_str_map[label_map[label]] = label

            # load shapes if given
            if FLAGS.shapes != '':
                with open(FLAGS.shapes, 'r') as f:
                    for line in f.readlines():
                        shape, idx = line.strip().split("\t")
                        shape_map[shape] = int(idx)

            # load chars if given
            if FLAGS.chars != '':
                update_chars = FLAGS.update_maps
                with open(FLAGS.chars, 'r') as f:
                    for line in f.readlines():
                        char, idx = line.strip().split("\t")
                        char_map[char] = int(idx)
                        char_int_str_map[char_map[char]] = char

            num_tokens = 0
            num_sentences = 0
            num_oov = 0
            num_docs = 0

            # TODO refactor this!!!
            if FLAGS.dataset == "conll2003":
                if not os.path.exists(FLAGS.out_dir):
                    print("Output directory not found: %s" % FLAGS.out_dir)
                writer = tf.python_io.TFRecordWriter(FLAGS.out_dir + '/examples.proto')
                with open(FLAGS.in_file) as f:
                    line_buf = []
                    line = f.readline()
                    line_idx = 1
                    while line:
                        line = line.strip()

                        if FLAGS.documents:
                            if line.split(" ")[0] == DOC_MARKER:
                                if line_buf:
                                    # reached the end of a document; process the lines
                                    toks, oov, sent = make_example(writer, line_buf, label_map, token_map, shape_map, char_map, update_vocab, update_chars)
                                    num_tokens += toks
                                    num_oov += oov
                                    num_sentences += sent
                                    num_docs += 1
                                    line_buf = []
                            else:
                                # print(line)
                                line_buf.append(line)
                                line_idx += 1

                        else:
                            # if the line is not empty, add it to the buffer
                            if line:
                                line_buf.append(line)
                                line_idx += 1
                            # otherwise, if there's stuff in the buffer, process it
                            elif line_buf:
                                # reached the end of a sentence; process the line
                                toks, oov, sent = make_example(writer, line_buf, label_map, token_map, shape_map, char_map, update_vocab, update_chars)
                                num_tokens += toks
                                num_oov += oov
                                num_sentences += sent
                                line_buf = []
                        # print("reading line %d" % line_idx)
                        line = f.readline()
                    if line_buf:
                        make_example(writer, line_buf, label_map, token_map, shape_map, char_map, update_vocab, update_chars)
                writer.close()

                # export the string->int maps to file
                for f_str, id_map in [('label', label_map), ('token', token_map), ('shape', shape_map), ('char', char_map)]:
                    with open(FLAGS.out_dir + '/' + f_str + '.txt', 'w') as f:
                        [f.write(s + '\t' + str(i) + '\n') for (s, i) in id_map.items()]

                # export data sizes to file
                with open(FLAGS.out_dir + "/sizes.txt", 'w') as f:
                    print(num_sentences, file=f)
                    print(num_tokens, file=f)
                    print(num_docs, file=f)

            else:
                if not os.path.exists(FLAGS.out_dir):
                    print("Output directory not found: %s" % FLAGS.out_dir)
                if not os.path.exists(FLAGS.out_dir + "/protos"):
                    os.mkdir(FLAGS.out_dir + "/protos")
                for data_type in onto_genre:
                    num_tokens = 0
                    num_sentences = 0
                    num_oov = 0
                    num_docs = 0
                    writer = tf.python_io.TFRecordWriter('%s/protos/%s_examples.proto' % (FLAGS.out_dir, data_type))
                    file_list = [y for x in os.walk(FLAGS.in_file) for y in glob(os.path.join(x[0], '*_gold_conll')) \
                                 if "/" + data_type + "/" in y and "/english/" in y]

                    for f_path in file_list:

                        with open(f_path) as f:
                            line_buf = []
                            line = f.readline()
                            line_idx = 1
                            while line:
                                line = line.strip()

                                if FLAGS.documents:
                                    if line.startswith("#"):
                                        if line_buf:
                                            # reached the end of a document; process the lines
                                            toks, oov, sent = make_example(writer, line_buf, label_map, token_map, shape_map,
                                                                           char_map, update_vocab, update_chars)
                                            num_tokens += toks
                                            num_oov += oov
                                            num_sentences += sent
                                            num_docs += 1
                                            line_buf = []
                                    else:
                                        # print(line)
                                        if line_buf or (not line_buf and line):
                                            line_buf.append(line)
                                            line_idx += 1
                                else:
                                    # if the line is not empty, add it to the buffer
                                    if line:
                                        if not line.startswith("#"):
                                            line_buf.append(line)
                                            line_idx += 1
                                    # otherwise, if there's stuff in the buffer, process it
                                    elif line_buf:
                                        # reached the end of a sentence; process the line
                                        toks, oov, sent = make_example(writer, line_buf, label_map, token_map, shape_map, char_map,
                                                                       update_vocab, update_chars)
                                        num_tokens += toks
                                        num_oov += oov
                                        num_sentences += sent
                                        line_buf = []
                                # print("reading line %d" % line_idx)
                                line = f.readline()
                            if line_buf:
                                make_example(writer, line_buf, label_map, token_map, shape_map, char_map, update_vocab,
                                             update_chars)
                            # print("Processed %d sentences" % num_sentences)
                    writer.close()

                    # export the string->int maps to file
                    for f_str, id_map in [('label', label_map), ('token', token_map), ('shape', shape_map), ('char', char_map)]:
                        with open("%s/%s.txt" % (FLAGS.out_dir, f_str), 'w') as f:
                            [f.write(s + '\t' + str(i) + '\n') for (s, i) in id_map.items()]

                    # export data sizes to file
                    with open("%s/%s_sizes.txt" % (FLAGS.out_dir, data_type), 'w') as f:
                        print(num_sentences, file=f)
                        print(num_tokens, file=f)
                        print(num_docs, file=f)

            print("Embeddings coverage: %2.2f%%" % ((1 - (num_oov / num_tokens)) * 100))

        def preprocess():
            if FLAGS.out_dir == '':
                print('Must supply out_dir')
                sys.exit(1)
            tsv_to_examples()

        for file_in, file_lample in zip(self.file_list_in, self.file_list_lample):
            self.del_all_flags()
            print('infile' + file_in)
            tf.app.flags.DEFINE_string('in_file', file_in, 'tsv file containing string data')

            tf.app.flags.DEFINE_string('vocab', self.vocabs, 'file containing vocab (empty means make new vocab)')

            tf.app.flags.DEFINE_string('labels', '', 'file containing labels (but always add new labels)')
            tf.app.flags.DEFINE_string('shapes', '', 'file containing shapes (add new shapes only when adding new vocab)')
            tf.app.flags.DEFINE_string('chars', '', 'file containing characters')

            tf.app.flags.DEFINE_string('embeddings', self.embedding, 'pretrained embeddings')
            print('out_dir' + file_lample)
            tf.app.flags.DEFINE_string('out_dir', file_lample, 'export tf protos')

            tf.app.flags.DEFINE_integer('window_size', 3, 'window size (for computing padding)')
            tf.app.flags.DEFINE_boolean('lowercase', False, 'whether to lowercase')

            tf.app.flags.DEFINE_boolean('start_end', False, 'whether to use distinct start/end padding')
            tf.app.flags.DEFINE_boolean('debug', False, 'print debugging output')

            tf.app.flags.DEFINE_boolean('predict_pad', False, 'whether to predict padding labels')

            tf.app.flags.DEFINE_boolean('documents', False, 'whether to grab documents rather than sentences')

            tf.app.flags.DEFINE_boolean('update_maps', True, 'whether to update maps')
            print('vocab' + self.vocabs)
            tf.app.flags.DEFINE_string('update_vocab', '', 'file to update vocab with tokens from training data')
            tf.app.flags.DEFINE_string('dataset', 'conll2003', 'which dataset')
            tf.app.flags.DEFINE_string('f', '', 'kernel')
            FLAGS = tf.app.flags.FLAGS

            ZERO_STR = "<ZERO>"
            PAD_STR = "<PAD>"
            OOV_STR = "<OOV>"
            NONE_STR = "<NONE>"
            SENT_START = "<S>"
            SENT_END = "</S>"

            pad_strs = [PAD_STR, SENT_START, SENT_END, ZERO_STR, NONE_STR]

            DOC_MARKER_CONLL = "-DOCSTART-"
            DOC_MARKER_ONTONOTES = "#begin document"
            print(FLAGS.flag_values_dict())
            DOC_MARKER = DOC_MARKER_CONLL if FLAGS.dataset == "conll2003" else DOC_MARKER_ONTONOTES

            label_int_str_map = {}
            token_int_str_map = {}
            char_int_str_map = {}

            pad_width = int(FLAGS.window_size / 2)
            onto_genre = ["bn", "bc", "nw", "mz", "tc", "wb"]

            print('start preprocess')
            preprocess()
        print('start train')
        self.del_all_flags()
        tf.app.flags.DEFINE_string('train_dir', self.file_list_lample[0], 'directory containing preprocessed training data')
        # predict
        tf.app.flags.DEFINE_string('dev_dir', self.file_list_lample[1], 'directory containing preprocessed dev data')
        tf.app.flags.DEFINE_string('test_dir', '', 'directory containing preprocessed test data')
        tf.app.flags.DEFINE_string('maps_dir', self.file_list_lample[0], 'directory containing data intmaps')
        # tf.app.flags.DEFINE_string('load_model', './models', '')
        tf.app.flags.DEFINE_string('model_dir', self.model_path, 'save model to this dir (if empty do not save)')
        # predict
        tf.app.flags.DEFINE_string('load_dir', '', 'load model from this dir (if empty do not load)')

        tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use')
        tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')
        tf.app.flags.DEFINE_string('model', 'cnn', 'which model to use [cnn, seq2seq, lstm, bilstm]')
        tf.app.flags.DEFINE_integer('filter_size', 3, "filter size")
        #     tf.app.flags.DEFINE_string('initialization','identity','')
        tf.app.flags.DEFINE_float('lr', 0.0005, 'learning rate')
        tf.app.flags.DEFINE_float('l2', 0.0, 'l2 penalty')
        tf.app.flags.DEFINE_float('beta1', 0.9, 'beta1')
        tf.app.flags.DEFINE_float('beta2', 0.9, 'beta2')
        tf.app.flags.DEFINE_float('epsilon', 1e-6, 'epsilon')

        tf.app.flags.DEFINE_float('hidden_dropout', 0.85, 'hidden layer dropout rate')
        tf.app.flags.DEFINE_float('input_dropout', 0.65, 'input layer (word embedding) dropout rate')
        tf.app.flags.DEFINE_float('middle_dropout', 1.0, 'middle layer dropout rate')
        tf.app.flags.DEFINE_float('word_dropout', 0.85, 'whole-word (-> oov) dropout rate')

        tf.app.flags.DEFINE_float('clip_norm', 5, 'clip gradients to have norm <= this')
        tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
        tf.app.flags.DEFINE_integer('lstm_dim', 300, 'lstm internal dimension')
        tf.app.flags.DEFINE_integer('embed_dim', 100, 'word embedding dimension')
        tf.app.flags.DEFINE_integer('shape_dim', 5, 'shape embedding dimension')
        tf.app.flags.DEFINE_integer('char_dim', 0, 'character embedding dimension')
        tf.app.flags.DEFINE_integer('char_tok_dim', 0, 'character token embedding dimension')
        tf.app.flags.DEFINE_string('char_model', 'lstm', 'character embedding model (lstm, cnn)')

        tf.app.flags.DEFINE_integer('max_finetune_epochs', 2, 'train for this many epochs')
        tf.app.flags.DEFINE_integer('max_context_epochs', 2, 'train for this many epochs')

        tf.app.flags.DEFINE_integer('max_epochs', 2, 'train for this many epochs')
        #     tf.app.flags.DEFINE_integer('num_filters', 300, '')
        tf.app.flags.DEFINE_integer('log_every', 2, 'log status every k steps')
        tf.app.flags.DEFINE_string('embeddings', self.embedding, 'file of pretrained embeddings to use')
        tf.app.flags.DEFINE_string('nonlinearity', 'relu', 'nonlinearity function to use (tanh, sigmoid, relu)')
        tf.app.flags.DEFINE_boolean('until_convergence', False, 'whether to run until convergence')
        #     for predict
        tf.app.flags.DEFINE_boolean('evaluate_only', False, 'whether to only run evaluation')
        tf.app.flags.DEFINE_string('layers', "{'conv1': {'dilation': 1, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': false}, 'conv2': {'dilation': 2, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': false}, 'conv3': {'dilation': 1, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': true}}",
                                   'json definition of layers (dilation, filters, width)')
        tf.app.flags.DEFINE_string('print_preds', 'labels.txt', 'print out predictions (for conll eval script) to given file (or do not if empty)')
        tf.app.flags.DEFINE_boolean('viterbi', False, 'whether to use viberbi inference')
        #     for predict
        tf.app.flags.DEFINE_boolean('train_eval', True, 'whether to report train accuracy')
        tf.app.flags.DEFINE_boolean('memmap_train', True, 'whether to load all training examples into memory')
        tf.app.flags.DEFINE_boolean('projection', True, 'whether to do final halving projection (front end)')

        tf.app.flags.DEFINE_integer('block_repeats', 1, 'number of times to repeat the stacked dilations block')
        tf.app.flags.DEFINE_boolean('share_repeats', True, 'whether to share parameters between blocks')

        tf.app.flags.DEFINE_string('loss', 'mean', '')
        tf.app.flags.DEFINE_float('margin', 0.0, 'margin')

        tf.app.flags.DEFINE_float('char_input_dropout', 1.0, 'dropout for character embeddings')

        tf.app.flags.DEFINE_float('save_min', 0.0, 'min accuracy before saving')

        tf.app.flags.DEFINE_boolean('start_end', False, 'whether using start/end or just pad between sentences')
        tf.app.flags.DEFINE_float('regularize_drop_penalty', 1e-4, 'penalty for dropout regularization')

        tf.app.flags.DEFINE_boolean('documents', False, 'whether each example is a document (default: sentence)')
        tf.app.flags.DEFINE_boolean('ontonotes', False, 'evaluate each domain of ontonotes seperately')
        self.cnn_predict()

    def predict(self, data, *args, **kwargs):
        """
        this method is for predicting the sentences in the data folder. This method should be invoked after invoking
        read dataset and train
        :param data: the path of the test file
        :param args: 
        :param kwargs: 
        :return: 
        """
        self.del_all_flags()
        tf.app.flags.DEFINE_string('train_dir', self.file_list_lample[0], 'directory containing preprocessed training data')
        # predict
        tf.app.flags.DEFINE_string('dev_dir', self.file_list_lample[1], 'directory containing preprocessed dev data')
        tf.app.flags.DEFINE_string('test_dir', data, 'directory containing preprocessed test data')
        tf.app.flags.DEFINE_string('maps_dir', self.file_list_lample[0], 'directory containing data intmaps')
        # tf.app.flags.DEFINE_string('load_model', './models', '')
        tf.app.flags.DEFINE_string('model_dir', self.model_path, 'save model to this dir (if empty do not save)')
        # predict
        tf.app.flags.DEFINE_string('load_dir', self.model_path, 'load model from this dir (if empty do not load)')

        tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use')
        tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')
        tf.app.flags.DEFINE_string('model', 'cnn', 'which model to use [cnn, seq2seq, lstm, bilstm]')
        tf.app.flags.DEFINE_integer('filter_size', 3, "filter size")
        #     tf.app.flags.DEFINE_string('initialization','identity','')
        tf.app.flags.DEFINE_float('lr', 0.0005, 'learning rate')
        tf.app.flags.DEFINE_float('l2', 0.0, 'l2 penalty')
        tf.app.flags.DEFINE_float('beta1', 0.9, 'beta1')
        tf.app.flags.DEFINE_float('beta2', 0.9, 'beta2')
        tf.app.flags.DEFINE_float('epsilon', 1e-6, 'epsilon')

        tf.app.flags.DEFINE_float('hidden_dropout', 0.85, 'hidden layer dropout rate')
        tf.app.flags.DEFINE_float('input_dropout', 0.65, 'input layer (word embedding) dropout rate')
        tf.app.flags.DEFINE_float('middle_dropout', 1.0, 'middle layer dropout rate')
        tf.app.flags.DEFINE_float('word_dropout', 0.85, 'whole-word (-> oov) dropout rate')

        tf.app.flags.DEFINE_float('clip_norm', 5, 'clip gradients to have norm <= this')
        tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
        tf.app.flags.DEFINE_integer('lstm_dim', 300, 'lstm internal dimension')
        tf.app.flags.DEFINE_integer('embed_dim', 100, 'word embedding dimension')
        tf.app.flags.DEFINE_integer('shape_dim', 5, 'shape embedding dimension')
        tf.app.flags.DEFINE_integer('char_dim', 0, 'character embedding dimension')
        tf.app.flags.DEFINE_integer('char_tok_dim', 0, 'character token embedding dimension')
        tf.app.flags.DEFINE_string('char_model', 'lstm', 'character embedding model (lstm, cnn)')

        tf.app.flags.DEFINE_integer('max_finetune_epochs', 2, 'train for this many epochs')
        tf.app.flags.DEFINE_integer('max_context_epochs', 2, 'train for this many epochs')

        tf.app.flags.DEFINE_integer('max_epochs', 2, 'train for this many epochs')
        #     tf.app.flags.DEFINE_integer('num_filters', 300, '')
        tf.app.flags.DEFINE_integer('log_every', 2, 'log status every k steps')
        tf.app.flags.DEFINE_string('embeddings', self.embedding, 'file of pretrained embeddings to use')
        tf.app.flags.DEFINE_string('nonlinearity', 'relu', 'nonlinearity function to use (tanh, sigmoid, relu)')
        tf.app.flags.DEFINE_boolean('until_convergence', False, 'whether to run until convergence')
        #     for predict
        tf.app.flags.DEFINE_boolean('evaluate_only', True, 'whether to only run evaluation')
        tf.app.flags.DEFINE_string('layers', "{'conv1': {'dilation': 1, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': false}, 'conv2': {'dilation': 2, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': false}, 'conv3': {'dilation': 1, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': true}}", 'json definition of layers (dilation, filters, width)')
        tf.app.flags.DEFINE_string('print_preds', 'labels.txt', 'print out predictions (for conll eval script) to given file (or do not if empty)')
        tf.app.flags.DEFINE_boolean('viterbi', False, 'whether to use viberbi inference')
        #     for predict
        tf.app.flags.DEFINE_boolean('train_eval', True, 'whether to report train accuracy')
        tf.app.flags.DEFINE_boolean('memmap_train', True, 'whether to load all training examples into memory')
        tf.app.flags.DEFINE_boolean('projection', True, 'whether to do final halving projection (front end)')

        tf.app.flags.DEFINE_integer('block_repeats', 1, 'number of times to repeat the stacked dilations block')
        tf.app.flags.DEFINE_boolean('share_repeats', True, 'whether to share parameters between blocks')

        tf.app.flags.DEFINE_string('loss', 'mean', '')
        tf.app.flags.DEFINE_float('margin', 0.0, 'margin')

        tf.app.flags.DEFINE_float('char_input_dropout', 1.0, 'dropout for character embeddings')

        tf.app.flags.DEFINE_float('save_min', 0.0, 'min accuracy before saving')

        tf.app.flags.DEFINE_boolean('start_end', False, 'whether using start/end or just pad between sentences')
        tf.app.flags.DEFINE_float('regularize_drop_penalty', 1e-4, 'penalty for dropout regularization')

        tf.app.flags.DEFINE_boolean('documents', False, 'whether each example is a document (default: sentence)')
        tf.app.flags.DEFINE_boolean('ontonotes', False, 'evaluate each domain of ontonotes seperately')
        self.cnn_predict()
        with open('labels.txt', 'r') as f:
            result = '\n'.join(f.readlines())
        return result

    def evaluate(self, predictions=None, ground_truths=None, *args, **kwargs):
        """
        this function will retrieve the evaluation result of last predict invocation
        :param predictions: it is not used in this method 
        :param ground_truths: it is not used in this method
        :param args: 
        :param kwargs: 
        :return: 
        """
        if not os.path.exists('./eval.txt'):
            raise NameError('please run predict() before evaluate method')
        with open('./eval.txt', 'r') as f:
            x = f.readlines()
        return "".join(x)

    def cnn_predict(self):
        FLAGS = tf.app.flags.FLAGS
        train_dir = FLAGS.train_dir
        dev_dir = FLAGS.dev_dir
        maps_dir = FLAGS.maps_dir
        if train_dir == '':
            print('Must supply input data directory generated from tsv_to_tfrecords.py')
            sys.exit(1)

        with open(maps_dir + '/label.txt', 'r') as f:
            labels_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
            labels_id_str_map = {i: s for s, i in labels_str_id_map.items()}
            labels_size = len(labels_id_str_map)
        with open(maps_dir + '/token.txt', 'r') as f:
            vocab_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
            vocab_id_str_map = {i: s for s, i in vocab_str_id_map.items()}
            vocab_size = len(vocab_id_str_map)
        with open(maps_dir + '/shape.txt', 'r') as f:
            shape_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
            shape_id_str_map = {i: s for s, i in shape_str_id_map.items()}
            shape_domain_size = len(shape_id_str_map)
        with open(maps_dir + '/char.txt', 'r') as f:
            char_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
            char_id_str_map = {i: s for s, i in char_str_id_map.items()}
            char_domain_size = len(char_id_str_map)
        print("num classes: %d" % labels_size)
        size_files = [maps_dir + "/" + fname for fname in listdir(maps_dir) if fname.find("sizes") != -1]
        num_train_examples = 0
        num_tokens = 0
        for size_file in size_files:
            print(size_file)
            with open(size_file, 'r') as f:
                num_train_examples += int(f.readline()[:-1])
                num_tokens += int(f.readline()[:-1])
        print("num train examples: %d" % num_train_examples)
        print("num train tokens: %d" % num_tokens)
        dev_top_dir = '/'.join(dev_dir.split("/")[:-2]) if dev_dir.find("*") != -1 else dev_dir
        print(dev_top_dir)
        dev_size_files = [dev_top_dir + "/" + fname for fname in listdir(dev_top_dir) if fname.find("sizes") != -1]
        num_dev_examples = 0
        num_dev_tokens = 0
        for size_file in dev_size_files:
            print(size_file)
            with open(size_file, 'r') as f:
                num_dev_examples += int(f.readline()[:-1])
                num_dev_tokens += int(f.readline()[:-1])

        print("num dev examples: %d" % num_dev_examples)
        print("num dev tokens: %d" % num_dev_tokens)
        type_set = {}
        type_int_int_map = {}
        outside_set = ["O", "<PAD>", "<S>", "</S>", "<ZERO>"]
        for label, id in labels_str_id_map.items():
            label_type = label if label in outside_set else label[2:]
            if label_type not in type_set:
                type_set[label_type] = len(type_set)
            type_int_int_map[id] = type_set[label_type]
        print(type_set)

        # load embeddings, if given; initialize in range [-.01, .01]
        embeddings_shape = (vocab_size - 1, FLAGS.embed_dim)
        embeddings = tf_utils.embedding_values(embeddings_shape, old=False)
        embeddings_used = 0
        if FLAGS.embeddings != '':
            # embedding each word, totally 402623 tokens
            with open(FLAGS.embeddings, 'r') as f:
                for line in f.readlines():
                    split_line = line.strip().split(" ")
                    word = split_line[0]
                    embedding = split_line[1:]
                    if word in vocab_str_id_map:
                        embeddings_used += 1
                        # shift by -1 because we are going to add a 0 constant vector for the padding later
                        # embeddings[vocab_str_id_map[word] - 1] = map(float, embedding)
                        embeddings[vocab_str_id_map[word] - 1] = [float(x) for x in embedding]
                    elif word.lower() in vocab_str_id_map:
                        embeddings_used += 1
                        embeddings[vocab_str_id_map[word.lower()] - 1] = map(float, embedding)
        print("Loaded %d/%d embeddings (%2.2f%% coverage)" % (embeddings_used, vocab_size, embeddings_used / vocab_size * 100))

        layers_map = sorted(json.loads(FLAGS.layers.replace("'", '"')).items()) if FLAGS.model == 'cnn' else None

        pad_width = int(layers_map[0][1]['width'] / 2) if layers_map is not None else 1

        with tf.Graph().as_default():
            train_batcher = Batcher(train_dir, FLAGS.batch_size) if FLAGS.memmap_train else SeqBatcher(train_dir, FLAGS.batch_size)

            dev_batch_size = FLAGS.batch_size  # num_dev_examples
            dev_batcher = SeqBatcher(dev_dir, dev_batch_size, num_buckets=0, num_epochs=1)
            if FLAGS.ontonotes:
                domain_dev_batchers = {domain: SeqBatcher(dev_dir.replace('*', domain),
                                                          dev_batch_size, num_buckets=0, num_epochs=1)
                                       for domain in ['bc', 'nw', 'bn', 'wb', 'mz', 'tc']}

            train_eval_batch_size = FLAGS.batch_size
            train_eval_batcher = SeqBatcher(train_dir, train_eval_batch_size, num_buckets=0, num_epochs=1)

            char_embedding_model = BiLSTMChar(char_domain_size, FLAGS.char_dim, int(FLAGS.char_tok_dim / 2)) \
                if FLAGS.char_dim > 0 and FLAGS.char_model == "lstm" else \
                (CNNChar(char_domain_size, FLAGS.char_dim, FLAGS.char_tok_dim, layers_map[0][1]['width'])
                 if FLAGS.char_dim > 0 and FLAGS.char_model == "cnn" else None)
            char_embeddings = char_embedding_model.outputs if char_embedding_model is not None else None

            if FLAGS.model == 'cnn':
                model = CNN(
                    num_classes=labels_size,
                    vocab_size=vocab_size,
                    shape_domain_size=shape_domain_size,
                    char_domain_size=char_domain_size,
                    char_size=FLAGS.char_tok_dim,
                    embedding_size=FLAGS.embed_dim,
                    shape_size=FLAGS.shape_dim,
                    nonlinearity=FLAGS.nonlinearity,
                    layers_map=layers_map,
                    viterbi=FLAGS.viterbi,
                    projection=FLAGS.projection,
                    loss=FLAGS.loss,
                    margin=FLAGS.margin,
                    repeats=FLAGS.block_repeats,
                    share_repeats=FLAGS.share_repeats,
                    char_embeddings=char_embeddings,
                    embeddings=embeddings)
            elif FLAGS.model == "bilstm":
                model = BiLSTM(
                    num_classes=labels_size,
                    vocab_size=vocab_size,
                    shape_domain_size=shape_domain_size,
                    char_domain_size=char_domain_size,
                    char_size=FLAGS.char_dim,
                    embedding_size=FLAGS.embed_dim,
                    shape_size=FLAGS.shape_dim,
                    nonlinearity=FLAGS.nonlinearity,
                    viterbi=FLAGS.viterbi,
                    hidden_dim=FLAGS.lstm_dim,
                    char_embeddings=char_embeddings,
                    embeddings=embeddings)
            else:
                print(FLAGS.model + ' is not a valid model type')
                sys.exit(1)

            # Define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)

            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=FLAGS.epsilon, name="optimizer")

            model_vars = tf.global_variables()

            print("model vars: %d" % len(model_vars))
            print(list(map(lambda v: v.name, model_vars)))

            # todo put in func
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            print("Total trainable parameters: %d" % (total_parameters))

            if FLAGS.clip_norm > 0:
                grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, model_vars), FLAGS.clip_norm)
                train_op = optimizer.apply_gradients(zip(grads, model_vars), global_step=global_step)
            else:
                train_op = optimizer.minimize(model.loss, global_step=global_step, var_list=model_vars)

            tf.global_variables_initializer()

            opt_vars = [optimizer.get_slot(s, n) for n in optimizer.get_slot_names() for s in model_vars if optimizer.get_slot(s, n) is not None]
            model_vars += opt_vars

            if FLAGS.load_dir:
                reader = tf.train.NewCheckpointReader(FLAGS.load_dir + ".tf")
                saved_var_map = reader.get_variable_to_shape_map()
                intersect_vars = [k for k in tf.global_variables() if k.name.split(':')[0] in saved_var_map and k.get_shape() == saved_var_map[k.name.split(':')[0]]]
                leftovers = [k for k in tf.global_variables() if k.name.split(':')[0] not in saved_var_map or k.get_shape() != saved_var_map[k.name.split(':')[0]]]
                print("WARNING: Loading pretrained model, but not loading: ", map(lambda v: v.name, leftovers))
                loader = tf.train.Saver(var_list=intersect_vars)

            else:
                loader = tf.train.Saver(var_list=model_vars)

            saver = tf.train.Saver(var_list=model_vars)

            sv = tf.train.Supervisor(logdir=FLAGS.model_dir if FLAGS.model_dir != '' else None,
                                     global_step=global_step,
                                     saver=None,
                                     save_model_secs=0,
                                     save_summaries_secs=0)

            training_start_time = time.time()
            with sv.managed_session(FLAGS.master, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                def run_evaluation(eval_batches, extra_text=""):
                    predictions = []
                    for b, (eval_label_batch, eval_token_batch, eval_shape_batch, eval_char_batch, eval_seq_len_batch, eval_tok_len_batch, eval_mask_batch) in enumerate(eval_batches):
                        batch_size, batch_seq_len = eval_token_batch.shape

                        char_lens = np.sum(eval_tok_len_batch, axis=1)
                        max_char_len = np.max(eval_tok_len_batch)
                        eval_padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
                        for b in range(batch_size):
                            char_indices = [item for sublist in [range(i * max_char_len, i * max_char_len + d) for i, d in
                                                                 enumerate(eval_tok_len_batch[b])] for item in sublist]
                            eval_padded_char_batch[b, char_indices] = eval_char_batch[b][:char_lens[b]]

                        char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
                            char_embedding_model.input_chars: eval_padded_char_batch,
                            char_embedding_model.batch_size: batch_size,
                            char_embedding_model.max_seq_len: batch_seq_len,
                            char_embedding_model.token_lengths: eval_tok_len_batch,
                            char_embedding_model.max_tok_len: max_char_len
                        }

                        basic_feeds = {
                            model.input_x1: eval_token_batch,
                            model.input_x2: eval_shape_batch,
                            model.input_y: eval_label_batch,
                            model.input_mask: eval_mask_batch,
                            model.max_seq_len: batch_seq_len,
                            model.batch_size: batch_size,
                            model.sequence_lengths: eval_seq_len_batch
                        }

                        basic_feeds.update(char_embedding_feeds)
                        total_feeds = basic_feeds.copy()

                        if FLAGS.viterbi:
                            preds, transition_params = sess.run([model.predictions, model.transition_params], feed_dict=total_feeds)

                            viterbi_repad = np.empty((batch_size, batch_seq_len))
                            for batch_idx, (unary_scores, sequence_lens) in enumerate(zip(preds, eval_seq_len_batch)):
                                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params)
                                viterbi_repad[batch_idx] = viterbi_sequence
                            predictions.append(viterbi_repad)
                        else:
                            preds, scores = sess.run([model.predictions, model.unflat_scores], feed_dict=total_feeds)
                            predictions.append(preds)
                    if FLAGS.print_preds != '':
                        evaluation.print_conlleval_format(FLAGS.print_preds, eval_batches, predictions, labels_id_str_map, vocab_id_str_map, pad_width)

                    # print evaluation
                    f1_micro, precision = evaluation.segment_eval(eval_batches, predictions, type_set, type_int_int_map,
                                                                  labels_id_str_map, vocab_id_str_map,
                                                                  outside_idx=map(lambda t: type_set[t] if t in type_set else type_set["O"], outside_set),
                                                                  pad_width=pad_width, start_end=FLAGS.start_end,
                                                                  extra_text="Segment evaluation %s:" % extra_text)

                    return f1_micro, precision

                threads = tf.train.start_queue_runners(sess=sess)
                log_every = int(max(100, num_train_examples / 5))

                if FLAGS.load_dir != '':
                    print("Deserializing model: " + FLAGS.load_dir + ".tf")
                    loader.restore(sess, FLAGS.load_dir + ".tf")

                def get_dev_batches(seq_batcher):
                    batches = []
                    # load all the dev batches into memory
                    done = False
                    while not done:
                        try:
                            dev_batch = sess.run(seq_batcher.next_batch_op)
                            dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch, dev_tok_len_batch = dev_batch
                            mask_batch = np.zeros(dev_token_batch.shape)
                            actual_seq_lens = np.add(np.sum(dev_seq_len_batch, axis=1),
                                                     (2 if FLAGS.start_end else 1) * pad_width * (
                                                             (dev_seq_len_batch != 0).sum(axis=1) + (
                                                         0 if FLAGS.start_end else 1)))
                            for i, seq_len in enumerate(actual_seq_lens):
                                mask_batch[i, :seq_len] = 1
                            batches.append((dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch,
                                            dev_seq_len_batch, dev_tok_len_batch, mask_batch))
                        except:
                            done = True
                    return batches

                dev_batches = get_dev_batches(dev_batcher)
                if FLAGS.ontonotes:
                    domain_batches = {domain: get_dev_batches(domain_batcher)
                                      for domain, domain_batcher in domain_dev_batchers.iteritems()}

                train_batches = []
                if FLAGS.train_eval:
                    # load all the train batches into memory
                    done = False
                    while not done:
                        try:
                            train_batch = sess.run(train_eval_batcher.next_batch_op)
                            train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch = train_batch
                            mask_batch = np.zeros(train_token_batch.shape)
                            actual_seq_lens = np.add(np.sum(train_seq_len_batch, axis=1), (2 if FLAGS.start_end else 1) * pad_width * ((train_seq_len_batch != 0).sum(axis=1) + (0 if FLAGS.start_end else 1)))
                            for i, seq_len in enumerate(actual_seq_lens):
                                mask_batch[i, :seq_len] = 1
                            train_batches.append((train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch, mask_batch))
                        except Exception as e:
                            done = True
                if FLAGS.memmap_train:
                    train_batcher.load_and_bucket_data(sess)

                def train(max_epochs, best_score, model_hidden_drop, model_input_drop, until_convergence, max_lower=6, min_iters=20):
                    print("Training on %d sentences (%d examples)" % (num_train_examples, num_train_examples))
                    start_time = time.time()
                    train_batcher._step = 1.0
                    converged = False
                    examples = 0
                    log_every_running = log_every
                    epoch_loss = 0.0
                    num_lower = 0
                    training_iteration = 0
                    speed_num = 0.0
                    speed_denom = 0.0
                    while not sv.should_stop() and training_iteration < max_epochs and not (until_convergence and converged):
                        # evaluate
                        if examples >= num_train_examples:
                            training_iteration += 1

                            if FLAGS.train_eval:
                                run_evaluation(train_batches, "TRAIN (iteration %d)" % training_iteration)
                            print()
                            f1_micro, precision = run_evaluation(dev_batches, "TEST (iteration %d)" % training_iteration)
                            print("Avg training speed: %f examples/second" % (speed_num / speed_denom))

                            # keep track of running best / convergence heuristic
                            if f1_micro > best_score:
                                best_score = f1_micro
                                num_lower = 0
                                if FLAGS.model_dir != '' and best_score > FLAGS.save_min:
                                    save_path = saver.save(sess, FLAGS.model_dir + ".tf")
                                    print("Serialized model: %s" % save_path)
                            else:
                                num_lower += 1
                            if num_lower > max_lower and training_iteration > min_iters:
                                converged = True

                            # update per-epoch variables
                            log_every_running = log_every
                            examples = 0
                            epoch_loss = 0.0
                            start_time = time.time()

                        if examples > log_every_running:
                            speed_denom += time.time() - start_time
                            speed_num += examples
                            evaluation.print_training_error(examples, start_time, [epoch_loss], train_batcher._step)
                            log_every_running += log_every

                        # Training iteration

                        label_batch, token_batch, shape_batch, char_batch, seq_len_batch, tok_lengths_batch = \
                            train_batcher.next_batch() if FLAGS.memmap_train else sess.run(train_batcher.next_batch_op)

                        # make mask out of seq lens
                        batch_size, batch_seq_len = token_batch.shape

                        char_lens = np.sum(tok_lengths_batch, axis=1)
                        max_char_len = np.max(tok_lengths_batch)
                        padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
                        for b in range(batch_size):
                            char_indices = [item for sublist in [range(i * max_char_len, i * max_char_len + d) for i, d in
                                                                 enumerate(tok_lengths_batch[b])] for item in sublist]
                            padded_char_batch[b, char_indices] = char_batch[b][:char_lens[b]]

                        max_sentences = max(map(len, seq_len_batch))
                        new_seq_len_batch = np.zeros((batch_size, max_sentences))
                        for i, seq_len_list in enumerate(seq_len_batch):
                            new_seq_len_batch[i, :len(seq_len_list)] = seq_len_list
                        seq_len_batch = new_seq_len_batch
                        num_sentences_batch = np.sum(seq_len_batch != 0, axis=1)

                        mask_batch = np.zeros((batch_size, batch_seq_len)).astype("int")
                        actual_seq_lens = np.add(np.sum(seq_len_batch, axis=1), (2 if FLAGS.start_end else 1) * pad_width * (num_sentences_batch + (0 if FLAGS.start_end else 1))).astype('int')
                        for i, seq_len in enumerate(actual_seq_lens):
                            mask_batch[i, :seq_len] = 1
                        examples += batch_size

                        # apply word dropout
                        # create word dropout mask
                        word_probs = np.random.random(token_batch.shape)
                        drop_indices = np.where((word_probs > FLAGS.word_dropout) & (token_batch != vocab_str_id_map["<PAD>"]))
                        token_batch[drop_indices[0], drop_indices[1]] = vocab_str_id_map["<OOV>"]

                        char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
                            char_embedding_model.input_chars: padded_char_batch,
                            char_embedding_model.batch_size: batch_size,
                            char_embedding_model.max_seq_len: batch_seq_len,
                            char_embedding_model.token_lengths: tok_lengths_batch,
                            char_embedding_model.max_tok_len: max_char_len,
                            char_embedding_model.input_dropout_keep_prob: FLAGS.char_input_dropout
                        }

                        if FLAGS.model == "cnn":
                            cnn_feeds = {
                                model.input_x1: token_batch,
                                model.input_x2: shape_batch,
                                model.input_y: label_batch,
                                model.input_mask: mask_batch,
                                model.max_seq_len: batch_seq_len,
                                model.sequence_lengths: seq_len_batch,
                                model.batch_size: batch_size,
                                model.hidden_dropout_keep_prob: model_hidden_drop,
                                model.input_dropout_keep_prob: model_input_drop,
                                model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                model.l2_penalty: FLAGS.l2,
                                model.drop_penalty: FLAGS.regularize_drop_penalty,
                            }
                            cnn_feeds.update(char_embedding_feeds)
                            _, loss = sess.run([train_op, model.loss], feed_dict=cnn_feeds)
                        elif FLAGS.model == "bilstm":
                            lstm_feed = {
                                model.input_x1: token_batch,
                                model.input_x2: shape_batch,
                                model.input_y: label_batch,
                                model.input_mask: mask_batch,
                                model.sequence_lengths: seq_len_batch,
                                model.max_seq_len: batch_seq_len,
                                model.batch_size: batch_size,
                                model.hidden_dropout_keep_prob: FLAGS.hidden_dropout,
                                model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                model.input_dropout_keep_prob: FLAGS.input_dropout,
                                model.l2_penalty: FLAGS.l2,
                                model.drop_penalty: FLAGS.regularize_drop_penalty
                            }
                            lstm_feed.update(char_embedding_feeds)
                            _, loss = sess.run([train_op, model.loss], feed_dict=lstm_feed)
                        epoch_loss += loss
                        train_batcher._step += 1
                    return best_score, training_iteration, speed_num / speed_denom

                if FLAGS.evaluate_only:
                    if FLAGS.train_eval:
                        run_evaluation(train_batches, "(train)")
                    print()
                    run_evaluation(dev_batches, "(test)")
                    if FLAGS.ontonotes:
                        for domain, domain_batches in domain_batches.iteritems():
                            print()
                            run_evaluation(domain_batches, FLAGS.layers2 != '', "(test - domain: %s)" % domain)

                else:
                    best_score, training_iteration, train_speed = train(FLAGS.max_epochs, 0.0,
                                                                        FLAGS.hidden_dropout, FLAGS.input_dropout,
                                                                        until_convergence=FLAGS.until_convergence)
                    if FLAGS.model_dir:
                        print("Deserializing model: " + FLAGS.model_dir + ".tf")
                        saver.restore(sess, FLAGS.model_dir + ".tf")

                sv.coord.request_stop()
                sv.coord.join(threads)
                sess.close()

                total_time = time.time() - training_start_time
                if FLAGS.evaluate_only:
                    print("Testing time: %d seconds" % (total_time))
                else:
                    print("Training time: %d minutes, %d iterations (%3.2f minutes/iteration)" % (total_time / 60, training_iteration, total_time / (60 * training_iteration)))
                    print("Avg training speed: %f examples/second" % (train_speed))
                    print("Best dev F1: %2.2f" % (best_score * 100))
