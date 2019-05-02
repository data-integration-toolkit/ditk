import collections
import os
import re
from mner import modeling
from mner import optimization
from mner import tokenization
import tensorflow as tf

import tf_metrics
import pickle

from configure import FLAGS

from ner import Ner

from m_ner import MultiNer


def main(input_file_path):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf_model = MultiNer()

    file_dict = {
        "train": {
            "data": "./data/train.txt"
        },
        "dev": {
            "data": "./data/dev.txt"
        },
        "test": {
            "data": input_file_path
        }
    }

    train_data, test_data, dev_data = tf_model.read_dataset(file_dict, 'ditk')
    tf_model.train(train_data)

    predictions = my_model.predict(test_data)
    # print("prediction file path : {}".format(os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))))

    tf_model.evaluate(test_data)

    return os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))


if __name__ == "__main__":
    input_file_path = "/data/test.txt"
    output_file_path = main(input_file_path)
    print(output_file_path)

