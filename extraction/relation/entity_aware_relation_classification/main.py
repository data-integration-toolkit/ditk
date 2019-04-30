import os
import numpy as np
import pandas as pd
import tensorflow as tf
import copy

import data_helpers
from configure import FLAGS
from logger import Logger
from model import entity_att_lstm
import utils
import datetime

from relation_extraction import RelationExtraction

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

from lstm_relation_extraction import LSTM_relation_extraction


def main(input_file_path):
    my_model = LSTM_relation_extraction()
    FLAGS.embeddings = 'glove300'
    FLAGS.data_type = 'semeval2010'

    files_dict = {
        "train": "data/"+FLAGS.data_type+"/trainfile.txt",
        "test": input_file_path,
    }

    train_data, test_data = my_model.read_dataset(files_dict)

    predictions, output_file_path = my_model.predict(test_data)

    precision, recall, f1 = my_model.evaluate(None, predictions)

    return output_file_path


if __name__ == "__main__":
    main("data/semeval2010/testfile.txt")

