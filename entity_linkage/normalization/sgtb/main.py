import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from structured_gradient_tree_boosting import StructuredGradientTreeBoosting


def main(file_name):
    sgtb = StructuredGradientTreeBoosting()

    # read data
    ratio = (0.7, 0.1, 0.2)
    options = {}
    train_set, dev_set, test_set = sgtb.read_dataset(file_name, ratio, options)


    # train
    model = sgtb.train([train_set]+[dev_set])


    # save
    fileName = "./model/finalized_model.sav"
    sgtb.save_model(model, fileName)

    # load
    fileName = "./model/finalized_model.sav"
    model = sgtb.load_model(fileName)


    # predict
    result = sgtb.predict(model, test_set)

    # evaluation
    result = sgtb.evaluate(model, test_set)


if __name__ == "__main__":
    file_name = "./data/AIDA-PPR-processed-sample.json"
    main(file_name)