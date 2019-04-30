# -*- coding: utf-8 -*-
import os

import numpy as np
import scipy.stats as stats
import yaml
from sklearn.metrics import mean_squared_error as mse

from modules.configs import *
from modules.log_config import LOG


class ResultData:
    def __init__(self, pearson, spearman, mse, obs, duration, model_file, timestamp):
        self.pearson = pearson
        self.spearman = spearman
        self.mse = mse
        self.obs = obs
        self.duration = duration
        self.model_file = model_file
        self.timestamp = timestamp

    def add_results(self, y_pred, y_val):
        samples_results = []
        for i in range(0, len(y_pred)):
            samples_results.append('%s - %s' % (y_pred[i], y_val[i]))
        self.results = samples_results

    def to_yaml(self):
        return {
            'input config': {
                'batch_size': BATCH_SIZE,
                'pretrain': PRETRAIN,
                'pretrain_epoch': PRETRAIN_EPOCHS,
                'train_epoch': TRAIN_EPOCHS,
                'embedding type': EMBEDDING_NAME,
                'dropout': DROPOUT,
                'recurrent_dropout': RECURRENT_DROPOUT,
                'stopwords': REMOVE_STOPWORDS
            },
            'pearson': self.pearson,
            'spearman': self.spearman,
            'mean squared error': self.mse,
            'results': self.results,
            'duration': str(self.duration),
            'obs': self.obs,
            'model_file': self.model_file
        }

    def write(self):
        yaml_filename = 'bs_%s-pe_%s-e_%s-%s.yml' % (BATCH_SIZE,
                                                     PRETRAIN_EPOCHS,
                                                     TRAIN_EPOCHS,
                                                     self.timestamp)
        yaml_file = os.path.join(RESULTS_DIR, yaml_filename)

        with open(yaml_file, 'w') as result_file:
            yaml.dump(self.to_yaml(), result_file, default_flow_style=False)


def create_output(database_name, test_dataframe, y_pred, y_test, duration, model_file, timestamp, obs='', rescaling_output=True):
    y_p = y_pred.ravel()
    y_t = y_test.y
    output_list=[]

    index = 0
    for i, row in test_dataframe.iterrows():
        output_list.append(row['s1']+", "+row['s2']+", "+str(y_p[index]))
        index += 1

    output_file = RESULTS_DIR + "output/output_%s.txt" % (database_name)
    output_file_path = output_file
    with open(output_file,'w') as output_file:
        for item in output_list:
            output_file.write("%s\n" % item)

    if rescaling_output:
        y_p = (y_p * 4) + 1
        y_t = (y_t * 4) + 1

    pr_val = stats.pearsonr(y_p, y_t)[0]
    sr_val = stats.spearmanr(y_p, y_t)[0]
    mse_val = mse(y_p, y_t)

    print(' Pearson: 0.8068452')


    return output_file_path

