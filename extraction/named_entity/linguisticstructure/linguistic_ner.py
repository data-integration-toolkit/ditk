import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time
import random
import codecs
import ditk_converter_utils
import numpy as np
import tensorflow as tf

import rnn

batch_nodes = 500
batch_trees = 16


class LinguisticRNN():

    def __init__(self):
        self.word_list = None
        self.ne_list = None
        self.character_length = None
        self.pos_length = None
        self.entities_length = None
        self.lexicon_length = None
        self.dataset = None
        self.model = None
        self.conll2012 = False
        self.patience = 20
        self.max_epoches = 100
        self.senna_hash_path = None
        self.glove_path = None

    def process_dataset(self, data_dict, **kwargs):
        self.conll2012 = ditk_converter_utils.is_data_conll12(data_dict['train'])
        self.ne_tags = ditk_converter_utils.get_ne_tags(data_dict['train'] + data_dict['test'] + data_dict['dev'])
        self.pos_tags = ditk_converter_utils.get_pos_tags(data_dict['train'] + data_dict['test'] + data_dict['dev'])

        if 'embeddings' in kwargs:
            embeddings = kwargs['embeddings']
            if self.glove_path is None:
                if 'glove_path' in embeddings:
                    self.glove_path = embeddings['glove_path']
                else:
                    raise FileNotFoundError('Missing or incorrect glove file path. Input correct glove_path in embeddings input')

            if self.senna_hash_path is None:
                if 'senna_hash_path' in embeddings:
                    self.senna_hash_path = embeddings['senna_hash_path']
                else:
                    raise FileNotFoundError('Missing or incorrect senna hash path. Input correct sena_hash_path in embeddings input')
        else:
            if not self.senna_hash_path and not self.glove_path:
                raise ValueError('No embeddings loaded.')

        if self.conll2012:
            import ontonotes as data_utils
            self.dataset = 'ontonotes'
        else:
            import conll2003 as data_utils
            self.dataset = 'conll2003'

        data = {}
        for key in data_dict.keys():
            data[key] = self.convert_data(data_dict[key])
        data, self.word_list, self.ne_list, self.character_length, self.pos_length, self.entities_length, \
        self.lexicon_length = data_utils.preprocess_raw_dataset(data,ne_tags=self.ne_tags, pos_tags=self.pos_tags,senna_hash_path=self.senna_hash_path, glove_file_path=self.glove_path)
        return data

    def convert_data(self,data):
        pp_data = ditk_converter_utils.convert_to_line_format(data)
        pp_data = ditk_converter_utils.ditk_ner_to_conll2012(pp_data) if self.conll2012 else ditk_converter_utils.ditk_ner_to_conll2003(pp_data)
        data = ditk_converter_utils.convert_to_line_format(pp_data)
        return data

    def process_data(self, raw_data):
        data = self.convert_data(raw_data)
        if self.conll2012:
            import ontonotes as data_utils
        else:
            import conll2003 as data_utils

        data = data_utils.preprocess_data(data)
        return data

    def load_embedding(self):
        """ Load pre-trained word embeddings into the dictionary of model

        word.npy: an array of pre-trained words
        embedding.npy: a 2d array of pre-trained word vectors
        word_list: a list of words in the dictionary of model
        """
        # load pre-trained word embeddings from file
        word_array = np.load(os.path.join(self.dataset, "word.npy"))
        embedding_array = np.load(os.path.join(self.dataset, "embedding.npy"))
        word_to_embedding = {}
        for i, word in enumerate(word_array):
            word_to_embedding[word] = embedding_array[i]

        # Store pre-trained word embeddings into the dictionary of model
        L = self.model.sess.run(self.model.L)

        for index, word in enumerate(self.word_list):
            if word in word_to_embedding:
                L[index] = word_to_embedding[word]
        self.model.sess.run(self.model.L.assign(L))
        return

    def initialize_model(self, use_pretrained_embedding=True):
        """ Get tree data and initialize a model

        #data: a dictionary; key-value example: "train"-(tree_list, ner_list)
        data: a dictionary; key-value example:
            "train"-{"tree_pyramid_list": tree_pyramid_list, "ner_list": ner_list}
        tree_pyramid_list: a list of (tree, pyramid) tuples
        ner_list: a list of dictionaries; key-value example: (3,5)-"PERSON"
        ne_list: a list of distinct string labels, e.g. "PERSON"
        """
        if self.model and self.model.init:
            return

        # Load data and determine dataset related hyperparameters
        config = rnn.Config()

        config.alphabet_size = self.character_length
        config.pos_dimension = self.pos_length
        config.output_dimension = self.entities_length
        config.lexicons = self.lexicon_length
        config.vocabulary_size = len(self.word_list)

        # Initialize a model
        self.model = rnn.RNN(config)
        self.model.sess = tf.Session()
        self.model.sess.run(tf.global_variables_initializer())

        if use_pretrained_embedding: self.load_embedding()

    def make_batch_list(self, tree_pyramid_list):
        """ Create a list of batches of (tree, pyramid) tuples

        The (tree, pyramid) tuples in the same batch have similar numbers of nodes,
        so later padding can be minimized.
        """
        index_tree_pyramid_list = sorted(enumerate(tree_pyramid_list),
                                         key=lambda x: x[1][0].nodes + len(x[1][1]))

        batch_list = []
        batch = []
        for index, tree_pyramid in index_tree_pyramid_list:
            nodes = tree_pyramid[0].nodes + len(tree_pyramid[1])
            if len(batch) + 1 > batch_trees or (len(batch) + 1) * nodes > batch_nodes:
                batch_list.append(batch)
                batch = []
            batch.append((index, tree_pyramid))
        batch_list.append(batch)

        random.shuffle(batch_list)
        # batch_list = batch_list[::-1]
        return batch_list

    def train_an_epoch(self, tree_pyramid_list):
        """ Update model parameters for every tree once
        """
        batch_list = self.make_batch_list(tree_pyramid_list)

        total_trees = len(tree_pyramid_list)
        trees = 0
        loss = 0.
        for i, batch in enumerate(batch_list):
            _, tree_pyramid_list = zip(*batch)
            # print "YOLO %d %d" % (tree_pyramid_list[-1][0].nodes, len(tree_pyramid_list[-1][1]))
            loss += self.model.train(tree_pyramid_list)
            trees += len(batch)
            sys.stdout.write("\r(%5d/%5d) average loss %.3f   " % (trees, total_trees, loss / trees))
            sys.stdout.flush()

        sys.stdout.write("\r" + " " * 64 + "\r")
        return loss / total_trees

    def predict_dataset(self, tree_pyramid_list):
        """ Get dictionaries of predicted positive spans and their labels for every tree
        """
        batch_list = self.make_batch_list(tree_pyramid_list)

        ner_list = [None] * len(tree_pyramid_list)
        for batch in batch_list:
            index_list, tree_pyramid_list = zip(*batch)
            for i, span_y in enumerate(self.model.predict(tree_pyramid_list)):
                ner_list[index_list[i]] = {span: self.ne_list[y] for span, y in span_y.items()}
        return ner_list

    def evaluate_prediction(self, ner_list, ner_hat_list):
        """ Compute the score of the prediction of trees
        """
        reals = 0.
        positives = 0.
        true_positives = 0.
        for index, ner in enumerate(ner_list):
            ner_hat = ner_hat_list[index]
            reals += len(ner)
            positives += len(ner_hat)
            for span in ner_hat.keys():
                if span not in ner: continue
                if ner[span] == ner_hat[span]:
                    true_positives += 1

        try:
            precision = true_positives / positives
        except ZeroDivisionError:
            precision = 1.

        try:
            recall = true_positives / reals
        except ZeroDivisionError:
            recall = 1.

        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.

        return precision * 100, recall * 100, f1 * 100

    def evaluate_ditk(self, actual, predicted):
        """ Compute the score of the prediction
        """

        f1 = 0.
        tp = 0.
        tn = 0.
        fn = 0.
        fp = 0.

        for index, ner in enumerate(predicted):
            if len(actual[index]) == 0 and len(predicted[index]) == 0:
                tn += 1
            elif len(actual[index]) == 0 and len(predicted[index]) != 0:
                fn += 1
            elif actual[index][3] == 'O':
                if actual[index][3] == predicted[index][3]:
                    tn += 1
                else:
                    fp += 1
            elif actual[index][3] == predicted[index][3]:
                tp += 1

        try:
            precision = tp / (tp+fp)
        except ZeroDivisionError:
            precision = 1.

        try:
            recall = tp / (tp+fn)
        except ZeroDivisionError:
            recall = 0.

        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.

        return precision * 100, recall * 100, f1 * 100

    def train_model(self, data, pretrain=True, **kwargs):
        """ Update model parameters until it converges or reaches maximum epochs
        """
        data = self.process_dataset(data)
        self.initialize_model()

        if 'max_epoches' in kwargs: self.max_epoches = kwargs['max_epoches']
        if 'patience' in kwargs: self.patience = kwargs['patience']

        saver = tf.train.Saver()

        if pretrain:
            saver.restore(self.model.sess, "models/" + self.dataset + ".model")

        best_epoch = 0
        best_score = (-1, -1, -1)
        best_loss = float("inf")
        for epoch in range(1, self.max_epoches + 1):
            print("\n<Epoch %d>" % epoch)

            start_time = time.time()
            # Change back to data["train"]
            loss = self.train_an_epoch(data["dev"]["tree_pyramid_list"])
            print("[train] average loss %.3f; elapsed %.0fs" % (loss, time.time() - start_time))

            start_time = time.time()
            ner_hat_list = self.predict_dataset(data["dev"]["tree_pyramid_list"])
            score = self.evaluate_prediction(data["dev"]["ner_list"], ner_hat_list)
            print("[dev] precision=%.1f%% recall=%.1f%% f1=%.3f%%; elapsed %.0fs;" % (
                        score + (time.time() - start_time,)), )

            if best_score[2] < score[2]:
                print("best")
                best_epoch = epoch
                best_score = score
                best_loss = loss
                saver.save(self.model.sess, "models/" + self.dataset + ".model")
            else:
                print("worse #%d" % (epoch - best_epoch))
            if epoch - best_epoch >= self.patience: break

        print("\n<Best Epoch %d>" % best_epoch)
        print("[train] average loss %.3f" % best_loss)
        print("[dev] precision=%.1f%% recall=%.1f%% f1=%.3f%%" % best_score)

        # saver.restore(self.model.sess, "models/" + self.dataset + ".model")
        # ner_hat_list = self.predict_dataset(data["test"]["tree_pyramid_list"])
        # score = self.evaluate_prediction(data["test"]["ner_list"], ner_hat_list)
        # print("[test] precision=%.1f%% recall=%.1f%% f1=%.3f%%" % score)
        return

    def ner_diff(self, ner_a_list, ner_b_list):
        """
        Compute the differences of two ner predictions

        ner_list: a list of the ner prediction of each sentence
        ner: a dict of span-ne pairs
        """
        sentences = len(ner_a_list)
        print("%d sentences" % sentences)
        print("a: %d nes" % sum(len(ner) for ner in ner_a_list))
        print("b: %d nes" % sum(len(ner) for ner in ner_b_list))

        ner_aa_list = []
        ner_bb_list = []
        ner_ab_list = []
        for i in range(sentences):
            ner_aa = {span: ne for span, ne in ner_a_list[i].items()}
            ner_bb = {span: ne for span, ne in ner_b_list[i].items()}
            ner_ab = {}
            for span, ne in ner_aa.items():
                if span in ner_bb and ner_aa[span] == ner_bb[span]:
                    del ner_aa[span]
                    del ner_bb[span]
                    ner_ab[span] = ne
            ner_aa_list.append(ner_aa)
            ner_bb_list.append(ner_bb)
            ner_ab_list.append(ner_ab)

        return ner_aa_list, ner_bb_list, ner_ab_list

    def write_ner(self, target_file, text_raw_data, ner_list):
        """
        Write the ner prediction of each sentence to file,
        indexing the senteces from 0.

        ner_list: a list of the ner prediction of each sentence
        ner: a dict of span-ne pairs
        """
        print("")
        print(target_file)
        sentences = len(text_raw_data)
        print("%d sentences" % sentences)

        with codecs.open(target_file, "w", encoding="utf8") as f:
            for i in range(sentences):
                if len(ner_list) == 0 or len(ner_list[i]) == 0:
                    for word in text_raw_data[i]:
                        f.write("%s %s %s\n" % (word[0],word[1], 'O'))
                else:
                    prev = 0
                    for span, ne in ner_list[i].items():
                        non_ner = text_raw_data[i][prev:span[0]]
                        ner = text_raw_data[i][span[0]:span[1]]
                        for word in non_ner:
                            f.write("%s %s %s\n" % (word[0],word[1], 'O'))

                        flag = False
                        for word in ner:
                            if not flag: tag = 'B-'
                            else: tag = 'I-'
                            f.write("%s %s %s\n" % (word[0],word[1], tag+ne))
                            flag = True
                        prev = span[1]

                        non_ner = text_raw_data[i][prev:]
                        for word in non_ner:
                            f.write("%s %s %s\n" % (word[0],word[1], 'O'))
                f.write('\n')

        print("%d nes" % sum(len(ner) for ner in ner_list))
        return

    def format_predictions(self, text_raw_data, ner_list):
        """
        Write the ner prediction of each sentence to file,
        indexing the senteces from 0.

        ner_list: a list of the ner prediction of each sentence
        ner: a dict of span-ne pairs
        """
        sentences = len(text_raw_data)
        formatted_predictions = []

        for i in range(sentences):
            if len(ner_list) == 0:
                for word in text_raw_data[i]:
                    formatted_predictions.append((None, None, word[0], 'O'))
            else:
                prev = 0
                for span, ne in ner_list[i].items():
                    non_ner = text_raw_data[i][prev:span[0]]
                    ner = text_raw_data[i][span[0]:span[1]]
                    for word in non_ner:
                        formatted_predictions.append((None, None,word[0], 'O'))

                    flag = False
                    for word in ner:
                        if not flag: tag = 'B-'
                        else: tag = 'I-'
                        formatted_predictions.append((None, None,word[0], tag+ne))
                        flag = True
                    prev = span[1]

                non_ner = text_raw_data[i][prev:]
                for word in non_ner:
                    formatted_predictions.append((None, None, word[0], 'O'))

            formatted_predictions.append(())

        return formatted_predictions

    def predict(self, raw_data):
        self.initialize_model(use_pretrained_embedding=True)
        try:
            saver = tf.train.Saver()
            saver.restore(self.model.sess, "models/" + self.dataset + ".model")
        except Exception:
            raise FileNotFoundError('Trained model not found at default location. Please load model.')

        sentence_data = ditk_converter_utils.get_raw_sentences(raw_data)
        data = self.process_data(raw_data)
        ner_hat_list = self.predict_dataset(data["tree_pyramid_list"])

        formatted_predictions = self.format_predictions(sentence_data, ner_hat_list)
        return formatted_predictions

    def evaluate(self, predicted, actual):
        self.write_output('ner_test_output.txt', actual,predicted)
        score = self.evaluate_ditk(actual, predicted)
        return score

    def save_model(self, location='models'):
        saver = tf.train.Saver()
        if not self.model:
            print('No trained model in memory')
        else:
            saver.save(self.model.sess, location + "/" + self.dataset + ".model")

    def load_model(self, location='models'):
        self.initialize_model()
        saver = tf.train.Saver()
        try:
            saver.restore(self.model.sess, location+"/" + self.dataset + ".model")
        except Exception:
            raise FileNotFoundError("Error loading the model. Please verify the files in location.")

    def get_inverted_index(self, data):
        line_to_index = {line: index for index, line in enumerate(data)}
        return data, line_to_index

    def write_output(self, file, actual, predicted):
        with open(file, 'w') as f:
            for idx, sent in enumerate(predicted):
                if len(sent) == 0:
                    if idx+1 != len(predicted):
                        f.write('\n')
                else:
                    act_tag = actual[idx][3]
                    pred_tag = sent[3]
                    word = sent[2]
                    f.write("%s %s %s\n" % (word, act_tag, pred_tag))
