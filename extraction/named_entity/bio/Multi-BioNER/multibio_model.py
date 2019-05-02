import argparse
import os
import functools
import time
from tqdm import tqdm
import torch
from model.lm_lstm_crf import *
from ner import Ner
import codecs
import model.utils as utils
from model.crf import *
from model.evaluator import eval_wc
from model.predictor import predict_wc #NEW
import random
import sys
import itertools
import json

class MultiBio(Ner):
    def __init__(self, args):
        self.file_num = 0
        self.lines = []
        self.dev_lines = []
        self.test_lines = []
        self.dataset_loader = []
        self.dev_dataset_loader = []
        self.test_dataset_loader = []
        self.f_map = dict()
        self.l_map = dict()
        self.char_count = dict()
        self.train_features = []
        self.dev_features = []
        self.test_features = []
        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []
        self.train_features_tot = []
        self.test_word = []
        self.test_word_tag = []
        self.track_list = list()
        self.args = args
        if self.args.gpu >= 0:
            torch.cuda.set_device(self.args.gpu)
        print('setting:')
        self.args.output_annotation = True
        self.args.caseless = True
        self.args.fine_tune = True
        self.args.shrink_embedding = True
        print(self.args)
    def convert_ground_truth(self, data, *args, **kwargs):
        pass

    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
        print('loading corpus')
        self.file_num = len(self.args.train_file)
        for i in range(self.file_num):
            with codecs.open(self.args.train_file[i], 'r', 'utf-8') as f:
                lines0 = f.readlines()
                lines0 = lines0[0:2000]
                # print (len(lines0))
            self.lines.append(lines0)
        for i in range(self.file_num):
            with codecs.open(self.args.dev_file[i], 'r', 'utf-8') as f:
                dev_lines0 = f.readlines()
                dev_lines0 = dev_lines0[0:2000]
            self.dev_lines.append(dev_lines0)
        for i in range(self.file_num):
            with codecs.open(self.args.test_file[i], 'r', 'utf-8') as f:
                test_lines0 = f.readlines()
                test_lines0 = test_lines0[0:2000]
            self.test_lines.append(test_lines0)

        for i in range(self.file_num):
            dev_features0, dev_labels0 = utils.read_corpus(self.dev_lines[i])
            test_features0, test_labels0 = utils.read_corpus(self.test_lines[i])

            self.dev_features.append(dev_features0)
            self.test_features.append(test_features0)
            self.dev_labels.append(dev_labels0)
            self.test_labels.append(test_labels0)

            if self.args.output_annotation:  # NEW
                test_word0, test_word_tag0 = utils.read_features(self.test_lines[i])
                self.test_word.append(test_word0)
                self.test_word_tag.append(test_word_tag0)
            #print (len(self.test_word), len(self.test_labels))
            if self.args.load_check_point:
                if os.path.isfile(self.args.load_check_point):
                    print("loading checkpoint: '{}'".format(self.args.load_check_point))
                    self.checkpoint_file = torch.load(self.args.load_check_point)
                    self.args.start_epoch = self.checkpoint_file['epoch']
                    self.f_map = self.checkpoint_file['f_map']
                    self.l_map = self.checkpoint_file['l_map']
                    c_map = self.checkpoint_file['c_map']
                    self.in_doc_words = self.checkpoint_file['in_doc_words']
                    self.train_features, self.train_labels = utils.read_corpus(self.lines[i])
                else:
                    print("no checkpoint found at: '{}'".format(self.args.load_check_point))
            else:
                print('constructing coding table')
                train_features0, train_labels0, self.f_map, self.l_map, self.char_count = utils.generate_corpus_char(self.lines[i], self.f_map,
                                                                                                      self.l_map, self.char_count,
                                                                                                      c_thresholds=self.args.mini_count,
                                                                                                      if_shrink_w_feature=False)
            self.train_features.append(train_features0)
            self.train_labels.append(train_labels0)

            self.train_features_tot += train_features0

        shrink_char_count = [k for (k, v) in iter(self.char_count.items()) if v >= self.args.mini_count]
        self.char_map = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}

        self.char_map['<u>'] = len(self.char_map)  # unk for char
        self.char_map[' '] = len(self.char_map)  # concat for char
        self.char_map['\n'] = len(self.char_map)  # eof for char

        f_set = {v for v in self.f_map}
        dt_f_set = f_set
        self.f_map = utils.shrink_features(self.f_map, self.train_features_tot, self.args.mini_count)
        l_set = set()

        for i in range(self.file_num):
            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), self.dev_features[i]), dt_f_set)
            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), self.test_features[i]), dt_f_set)

            l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), self.dev_labels[i]), l_set)
            l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), self.test_labels[i]), l_set)

        if not self.args.rand_embedding:
            print("feature size: '{}'".format(len(self.f_map)))
            print('loading embedding')
            if self.args.fine_tune:  # which means does not do fine-tune
                self.f_map = {'<eof>': 0}
            self.f_map, self.embedding_tensor, self.in_doc_words = utils.load_embedding_wlm(self.args.emb_file, ' ', self.f_map, dt_f_set,
                                                                             self.args.caseless, self.args.unk, self.args.word_dim,
                                                                             shrink_to_corpus=self.args.shrink_embedding)
            print("embedding size: '{}'".format(len(self.f_map)))

        for label in l_set:

            if label not in self.l_map:
                self.l_map[label] = len(self.l_map)

        print('constructing dataset')
        for i in range(self.file_num):
            # construct dataset
            dataset, forw_corp, back_corp = utils.construct_bucket_mean_vb_wc(self.train_features[i], self.train_labels[i], self.l_map,
                                                                              self.char_map, self.f_map, self.args.caseless)
            dev_dataset, forw_dev, back_dev = utils.construct_bucket_mean_vb_wc(self.dev_features[i], self.dev_labels[i], self.l_map,
                                                                                self.char_map, self.f_map, self.args.caseless)
            test_dataset, forw_test, back_test = utils.construct_bucket_mean_vb_wc(self.test_features[i], self.test_labels[i],
                                                                                   self.l_map, self.char_map, self.f_map,
                                                                                   self.args.caseless)
            self.dataset_loader.append(
                [torch.utils.data.DataLoader(tup, self.args.batch_size, shuffle=True, drop_last=False) for tup in dataset])
            self.dev_dataset_loader.append(
                [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset])
            self.test_dataset_loader.append(
                [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset])

    def train(self, data, *args, **kwargs):
        tot_length = sum(map(lambda t: len(t), self.dataset_loader))
        loss_list = []
        acc_list = []
        best_f1 = []
        for i in range(self.file_num):
            best_f1.append(float('-inf'))

        best_pre = []
        for i in range(self.file_num):
            best_pre.append(float('-inf'))

        best_rec = []
        for i in range(self.file_num):
            best_rec.append(float('-inf'))


        start_time = time.time()
        epoch_list = range(self.args.start_epoch, self.args.start_epoch + self.args.epoch)
        patience_count = 0
        for epoch_idx, self.args.start_epoch in enumerate(epoch_list):

            sample_num = 1

            epoch_loss = 0
            self.ner_model.train()

            for sample_id in tqdm(range(sample_num), mininterval=2,
                                  desc=' - Tot it %d (epoch %d)' % (tot_length, self.args.start_epoch), leave=False,
                                  file=sys.stdout):

                self.file_no = random.randint(0, self.file_num - 1)
                cur_dataset = self.dataset_loader[self.file_no]

                for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v in itertools.chain.from_iterable(cur_dataset):

                    f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v,
                                                                             len_v)

                    self.ner_model.zero_grad()
                    scores = self.ner_model(f_f, f_p, b_f, b_p, w_f, self.file_no)
                    loss = self.crit_ner(scores, tg_v, mask_v)

                    epoch_loss += utils.to_scalar(loss)
                    if self.args.co_train:
                        cf_p = f_p[0:-1, :].contiguous()
                        cb_p = b_p[1:, :].contiguous()
                        cf_y = w_f[1:, :].contiguous()
                        cb_y = w_f[0:-1, :].contiguous()
                        cfs, _ = self.ner_model.word_pre_train_forward(f_f, cf_p)
                        loss = loss + self.args.lambda0 * self.crit_lm(cfs, cf_y.view(-1))
                        cbs, _ = self.ner_model.word_pre_train_backward(b_f, cb_p)
                        loss = loss + self.args.lambda0 * self.crit_lm(cbs, cb_y.view(-1))
                    loss.backward()
                    nn.utils.clip_grad_norm(self.ner_model.parameters(), self.args.clip_grad)
                    self.optimizer.step()

            epoch_loss /= tot_length

            # update lr
            utils.adjust_learning_rate(self.optimizer, self.args.lr / (1 + (self.args.start_epoch + 1) * self.args.lr_decay))

            # eval & save check_point
            if 'f' in self.args.eva_matrix:
                dev_f1, dev_pre, dev_rec, dev_acc = self.evaluate(None, None, self.dev_dataset_loader[self.file_no],
                                                                         self.file_no)
                loss_list.append(epoch_loss)
                acc_list.append(dev_acc)
                if dev_f1 > best_f1[self.file_no]:
                    patience_count = 0
                    best_f1[self.file_no] = dev_f1
                    best_pre[self.file_no] = dev_pre
                    best_rec[self.file_no] = dev_rec
                    self.track_list.append(
                        {'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})
                    print('(loss: %.4f, epoch: %d, dataset: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f)' %
                          (epoch_loss,
                           self.args.start_epoch,
                           self.file_no,
                           dev_f1,
                           dev_pre,
                           dev_rec))
                    try:
                        self.save_model(None)
                    except Exception as inst:
                        print(inst)

                else:
                    patience_count += 1
                    print('(loss: %.4f, epoch: %d, dataset: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f)' %
                          (epoch_loss,
                           self.args.start_epoch,
                           self.file_no,
                           dev_f1,
                           dev_pre,
                           dev_rec))
                    self.track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})

            print('epoch: ' + str(self.args.start_epoch) + '\t in ' + str(self.args.epoch) + ' take: ' + str(
                time.time() - start_time) + ' s')

            if patience_count >= self.args.patience and self.args.start_epoch >= self.args.least_iters:
                break
        return loss_list, acc_list

    def predict(self, data, fout, file_no, **kwargs):
        """
        :param data:
        :param fout:
        :param file_no:
        :param kwargs:
        :return:
        """
        #print('annotating')
        with open(self.args.output_file + str(self.file_no) + '.txt', 'w') as fout:
            self.predictor.output_batch(self.ner_model, self.test_word[self.file_no], self.test_word_tag[self.file_no],fout, self.file_no)
        return self.args.output_file + str(self.file_no) + '.txt'

    def evaluate(self, predictions, groundTruths, dataset, file_no, **kwargs):
        return self.evaluator.calc_score(self.ner_model, self.test_dataset_loader[self.file_no], self.file_no)

    def save_model(self, file):
        #print("saving model")
        utils.save_checkpoint({
            'epoch': self.args.start_epoch,
            'state_dict': self.ner_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'f_map': self.f_map,
            'l_map': self.l_map,
            'c_map': self.char_map,
            'in_doc_words': self.in_doc_words
        }, {'track_list': self.track_list,
            'args': vars(self.args)
            }, self.args.checkpoint + 'cwlm_lstm_crf')

    def load_model(self, args):
        print("CSCI548 model loading")

        self.args = args
        print('loading dictionary')
        with open(self.args.load_arg, 'r') as f:
            jd = json.load(f)
        jd = jd['args']

        checkpoint_file = torch.load(self.args.load_check_point, map_location=lambda storage, loc: storage)
        f_map = checkpoint_file['f_map']
        l_map = checkpoint_file['l_map']
        c_map = checkpoint_file['c_map']
        in_doc_words = checkpoint_file['in_doc_words']

        # build model
        print('loading model')
        self.ner_model = LM_LSTM_CRF(len(l_map), len(c_map), jd['char_dim'], jd['char_hidden'], jd['char_layers'],
                                jd['word_dim'], jd['word_hidden'], jd['word_layers'], len(f_map), jd['drop_out'],
                                self.args.dataset_no, large_CRF=jd['small_crf'], if_highway=jd['high_way'],
                                in_doc_words=in_doc_words, highway_layers=jd['highway_layers'])

        self.ner_model.load_state_dict(checkpoint_file['state_dict'])

        if self.args.gpu >= 0:
            if_cuda = True
            torch.cuda.set_device(self.args.gpu)
            self.ner_model.cuda()
            packer = CRFRepack_WC(len(l_map), True)
        else:
            if_cuda = False
            packer = CRFRepack_WC(len(l_map), False)

        decode_label = (self.args.decode_type == 'label')
        predictor = predict_wc(if_cuda, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'],
                               l_map['<start>'],
                               decode_label, self.args.batch_size, jd['caseless'])
        evaluator = eval_wc(packer, l_map, self.args.eva_matrix)
        # loading corpus
        print('loading corpus')
        lines = []
        features = []
        tags = []
        feature_tags = []
        with codecs.open(self.args.input_file, 'r', 'utf-8') as f:
            for i, line in enumerate(f):
                #if i == 2000:
                   #break
                if line == '\r\n':
                    features.append(utils.read_features2(lines))
                    feature_tags.append(tags)
                    tags = []
                    lines = []
                    continue
                tmp = line.split(" ")
                lines.append(tmp[0])
                tags.append((tmp[3]))
        #print(len(feature_tags),len(features))
        for idx in range(self.args.dataset_no):
            print('annotating the entity type', idx)
            with open(self.args.output_file + str(idx) + '.txt', 'w') as fout:
                for feature, tag in zip(features, feature_tags):
                    predictor.output_batch(self.ner_model, feature, tag, fout, idx)
                    fout.write('\n')
        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(self.ner_model, self.test_dataset_loader[0], 0)
        print("Test evaluation: f1 = %.4f, recall = %.4f, precision = %.4f " % (test_f1, test_rec, test_pre))
        return self.args.output_file + str(idx) + '.txt'

    def build_model(self):
        print('building model')
        self.ner_model = LM_LSTM_CRF(len(self.l_map), len(self.char_map), self.args.char_dim, self.args.char_hidden, self.args.char_layers,
                                self.args.word_dim, self.args.word_hidden, self.args.word_layers, len(self.f_map), self.args.drop_out, self.file_num,
                                large_CRF=self.args.small_crf, if_highway=self.args.high_way, in_doc_words=self.in_doc_words,
                                highway_layers=self.args.highway_layers)

        if self.args.load_check_point:
            self.ner_model.load_state_dict(self.checkpoint_file['state_dict'])
        else:
            if not self.args.rand_embedding:
                self.ner_model.load_pretrained_word_embedding(self.embedding_tensor)
            self.ner_model.rand_init(init_word_embedding=self.args.rand_embedding)

        if self.args.update == 'sgd':
            self.optimizer = optim.SGD(self.ner_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.update == 'adam':
            self.optimizer = optim.Adam(self.ner_model.parameters(), lr=self.args.lr)

        if self.args.load_check_point and self.args.load_opt:
            self.optimizer.load_state_dict(self.checkpoint_file['optimizer'])

        self.crit_lm = nn.CrossEntropyLoss()
        self.crit_ner = CRFLoss_vb(len(self.l_map), self.l_map['<start>'], self.l_map['<pad>'])

        if self.args.gpu >= 0:
            if_cuda = True
            print('device: ' + str(self.args.gpu))
            torch.cuda.set_device(self.args.gpu)
            self.crit_ner.cuda()
            self.crit_lm.cuda()
            self.ner_model.cuda()
            self.packer = CRFRepack_WC(len(self.l_map), True)
        else:
            if_cuda = False
            self.packer = CRFRepack_WC(len(self.l_map), False)
        self.evaluator = eval_wc(self.packer, self.l_map, self.args.eva_matrix)

        self.predictor = predict_wc(if_cuda, self.f_map, self.char_map, self.l_map, self.f_map['<eof>'], self.char_map['\n'], self.l_map['<pad>'],
                               self.l_map['<start>'], True, self.args.batch_size, self.args.caseless)  # NEW
