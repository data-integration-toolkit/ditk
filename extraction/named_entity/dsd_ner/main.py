import numpy as np
import os
import copy
import sys

from configure import FLAGS

from ner import Ner

from dsd_ner import DSD_ner


def main(input_file_path):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf_model = DSD_ner(3)

    file_dict = {
        "train": {
            "data": "/data/train.txt"
        },
        "dev": {
            "data": "/data/dev.txt"
        },
        "test": {
            "data": input_file_path
        }
    }

    train_data = tf_model.read_dataset(file_dict)
    tf_model.train(train_data)
    gtm = tf_model.convert_ground_truth(train_data)
    predictions = tf_model.predict(train_data)
    # print("prediction file path : {}".format(os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))))

    precision, recall, f1 = tf_model.evaluate(predictions,gtm)

    output_file = os.path.dirname(train_file)
    output_file_path = os.path.join(output_file, "predictions.txt")

    with open(output_file_path, 'w') as f:
        for index, (g, p) in enumerate(zip(ground, predictions)):
            if len(g[3])==0:
                f.write("\n")
            else:
                f.write(g[2] + " " + g[3] + " " + p[3] + "\n")

    return output_file_path
    # return os.path.abspath(os.path.join(FLAGS.model_dir, "prediction.txt"))


if __name__ == "__main__":
    input_file_path = "/data/test.txt"
    output_file_path = main(input_file_path)
    print(output_file_path)

