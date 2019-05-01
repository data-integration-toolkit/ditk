# Script to predict and evaluate in a pipeline
__author__ = 'xiang'

import sys
from collections import  defaultdict
from evaluation import *
from emb_prediction import *

if __name__ == "__main__":

    # do prediction here
    _data = "NYT"
    _method = "request"
    _sim_func = "cosine"
    _threshold = 0.0

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    output = outdir +'/prediction_emb_' + _method + '_' + _sim_func + '.txt'
    ground_truth = load_labels(indir + '/mention_type_test.txt')

    ### Prediction
    predict(indir, outdir, _method, _sim_func, _threshold, output, None)

    ### Evluate embedding predictions
    predictions = load_labels(output)
    print 'Predicted labels (embedding):'

    none_label_index = find_none_index(indir + '/type.txt')
    print '%f\t%f\t%f\t' % evaluate_rm_neg(predictions, ground_truth, none_label_index)
