import csv
import unittest
import pandas as pd
import numpy as np
import json
import torch
from sqlnet.utils import *
from sqlnet.model.seq2sql import Seq2SQL
from sqlnet.model.sqlnet import SQLNet
import datetime
import argparse


class TestSQLNetMethods(unittest.TestCase):
    def loadargs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', action='store_true',
                            help='If set, train model')
        parser.add_argument('--test', action='store_true',
                            help='If set, run testing mode')
        parser.add_argument('--evaluate', action='store_true',
                            help='If set, run inference mode')
        parser.add_argument('--extract_emb', action='store_true',
                            help='If set, extract glove embedding for training')
        parser.add_argument('--toy', action='store_true',
                            help='If set, use small data; used for fast debugging.')
        parser.add_argument('--suffix', type=str, default='',
                            help='The suffix at the end of saved model name.')
        parser.add_argument('--ca', action='store_true',
                            help='Use conditional attention.')
        parser.add_argument('--dataset', type=int, default=0,
                            help='0: original dataset, 1: re-split dataset')
        parser.add_argument('--rl', action='store_true',
                            help='Use RL for Seq2SQL(requires pretrained model).')
        parser.add_argument('--baseline', action='store_true',
                            help='If set, then train Seq2SQL model; default is SQLNet model.')
        parser.add_argument('--train_emb', action='store_true',
                            help='Train word embedding for SQLNet(requires pretrained model).')
        self.args = parser.parse_args()
        self.args.toy = False
        self.args.dataset = 0

    def setUp(self):
        self.loadargs()
        self.method = "sqlnet"
        self.input_file = "testsample/sqlnet_test_input.csv"
        self.output_file = "testsample/sqlnet_test_output.csv"
        self.verificationErrors = []  # append exceptions for try-except errors

    def test_input_file_format(self):
        N_word = 300
        B_word = 42
        USE_SMALL = False

        test_sql_data, test_table_data, TEST_DB = load_dataset(
            args.dataset, use_small=USE_SMALL, dataset='test')

        self.assertIsNotNone(test_sql_data)
        self.assertIsNotNone(test_table_data)
        self.assertIsNotNone(TEST_DB)

        word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), \
                                 load_used=True, use_small=USE_SMALL)  # load_used can speed up loading

        self.assertIsNotNone(word_emb)

    def writetocsv(self, fname, entry):
        with open(fname, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerows(entry)

    def test_output(self):
        self.loadargs()

        N_word = 300
        B_word = 42
        USE_SMALL = False
        GPU = True
        BATCH_SIZE = 64
        TEST_ENTRY = (True, True, True)  # (AGG, SEL, COND)

        test_sql_data, test_table_data, TEST_DB = load_dataset(
            args.dataset, use_small=USE_SMALL, dataset='test')

        self.assertIsNotNone(test_sql_data)
        self.assertIsNotNone(test_table_data)
        self.assertIsNotNone(TEST_DB)

        word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), \
                                 load_used=True, use_small=USE_SMALL)  # load_used can speed up loading

        if args.baseline:
            model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU, trainable_emb=True)
        else:
            model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, gpu=GPU,
                           trainable_emb=True)

        self.assertIsNotNone(model)

        agg_m, sel_m, cond_m = best_model_name(args)
        print("Loading from %s" % agg_m)
        self.assertIsInstance(agg_m, str)
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print("Loading from %s" % sel_m)
        self.assertIsInstance(sel_m, str)
        model.sel_pred.load_state_dict(torch.load(sel_m))
        self.assertIsInstance(sel_m, str)
        print("Loading from %s" % cond_m)
        model.cond_pred.load_state_dict(torch.load(cond_m))
        self.assertIsInstance(cond_m, str)

        predicted_queries = []
        e4, e5 = epoch_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY, pred_queries_holder=predicted_queries)
        self.writetocsv('entry.csv', predicted_queries)
        self.assertIsInstance(e4, float)
        self.assertIsInstance(e5, float)
        print("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s" % (e4, e5))
        predicted_queries = []
        e6 = epoch_exec_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, pred_queries_holder=predicted_queries)
        self.writetocsv('db.csv', predicted_queries)
        self.assertIsInstance(e6, float)
        print("Test execution acc: %s" % e6)

        predicted_queries = []
        e4, e5 = epoch_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY, pred_queries_holder=predicted_queries)
        writetocsv('entry.csv', predicted_queries)
        print("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s" % (e4, e5))
        predicted_queries = []
        e6 = epoch_exec_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, pred_queries_holder=predicted_queries)
        writetocsv('db.csv', predicted_queries)
        print("Test execution acc: %s" % e6)
if __name__ == '__main__':
    unittest.main()
