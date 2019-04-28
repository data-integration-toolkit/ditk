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
    #find ditk_path from sys.path
    ditk_path = ""
    for path in sys.path:
        if "ditk" in path:
            ditk_path = path

    #instantiate the implemented class
    sgtb_module = StructuredGradientTreeBoosting()

    # read data
    ratio = (0.7, 0.1, 0.2)
    options = {}
    train_set, dev_set, test_set = sgtb_module.read_dataset(file_name, ratio, options)


    # train
    model = sgtb_module.train([train_set]+[dev_set])


    # save

    fileName = ditk_path+'/entity_linkage/normalization/sgtb/model/finalized_model.sav'
    sgtb_module.save_model(model, fileName)

    # load
    fileName = ditk_path+'/entity_linkage/normalization/sgtb/model/finalized_model.sav'
    model = sgtb_module.load_model(fileName)


    # predict
    result = sgtb_module.predict(model, test_set)

    # evaluation
    result = sgtb_module.evaluate(model, test_set)


if __name__ == "__main__":
    #find ditk_path from sys.path
    ditk_path = ""
    for path in sys.path:
        if "ditk" in path:
            ditk_path = path
    file_name = ditk_path+"/entity_linkage/normalization/sgtb/data/AIDA-PPR-processed-sample.json"
    main(file_name)