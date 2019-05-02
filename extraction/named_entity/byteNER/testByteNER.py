from collections import OrderedDict

import model, utils
from model import NERModel
import preprocess, postprocess
import keras.backend as K

import time
import os
import numpy as np
import cPickle as pkl
import sys

from sklearn.metrics import recall_score, precision_score, f1_score

from byteNER import byteNER

if __name__ == '__main__':

    file_dict = {
        "train": {"file 1":"examples/training.tsv"},
        "dev" : {"file 2":"examples/development.tsv"},
        "test" : {"file 3":"examples/evaluation.tsv"},
    }
    dataset_name = 'CONLL2003'
    # instatiate the class
    myModel = byteNER() 
    # read in a dataset for training
    data = myModel.read_dataset(file_dict, dataset_name)

    myModel.train(data,nb_epochs=1)  # trains the model and stores model state in object properties or similar
    
    predictions = myModel.predict(data)  # generate predictions! output format will be same for everyone
    test_labels = myModel.convert_ground_truth(data)  #<-- need ground truth labels need to be in same format as predictions!

    P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1
    print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
