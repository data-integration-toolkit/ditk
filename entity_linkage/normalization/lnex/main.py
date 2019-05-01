import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from LNEx import LNEx


def main(file_name):
    #find ditk_path from sys.path
    

    #instantiate the implemented class
    lnex_obj = LNEx()

    # read data
    test_set,eval_set = lnex_obj.read_dataset(file_name)


    # initialiaze gazetteer
    dummy = ""
    geo_info = lnex_obj.train(dummy)


    # predict
    lnex_obj.predict(geo_info, test_set)

    # evaluation
    lnex_obj.evaluate(geo_info, eval_set)


if __name__ == "__main__":
    #find ditk_path from sys.path
    ditk_path = ""
    for path in sys.path:
        if "ditk" in path:
            ditk_path = path
    print(ditk_path)
    file_name = ditk_path+"/entity_linkage/normalization/lnex/test/sample_tweets.txt"
    main(file_name)