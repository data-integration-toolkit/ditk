from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf

import tf_metrics
import pickle

from configure import FLAGS

from ner import Ner

from bert_ner import BERT_Ner


def main(input_file_path):
    tf.logging.set_verbosity(tf.logging.INFO)

    my_model = BERT_Ner()

    file_dict = {
        "train": {
            "data": input_file_path
        },
        "dev": {
            "data": "./NERdata/dev.txt"
        },
        "test": {
            "data": input_file_path
        }
    }

    train_data, test_data, dev_data = my_model.read_dataset(file_dict, 'ditk')
    my_model.train(train_data)

    tokens, prediction, labels = my_model.predict(test_data)
    print("prediction file path : {}".format(os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))))

    my_model.evaluate(dev_data)

    return os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))


if __name__ == "__main__":
    print(main("NERdata/test.txt"))

