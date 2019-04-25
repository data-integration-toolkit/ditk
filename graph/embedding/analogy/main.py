# import os
# from graph.embedding.analogy.analogy import ANALOGY

import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from analogy import ANALOGY

INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\FB15k\\"

# input_file_path = ""
# output_file_path = main(input_file_path)



def main(input_file_path):
    # output_file_path = ""

    print("Main Script")

    algorithm = ANALOGY()

    train_file_names = {"train": input_file_path + "train.txt",
                        # "valid": input_file_path + "valid.txt",
                        # "whole": input_file_path + "whole.txt",
                        "relations": input_file_path + "relation2id.txt",
                        "entities": input_file_path + "entity2id.txt"}

    algorithm.read_dataset(train_file_names)

    parameters = {"mode": 'single',
            "epoch": 3,
            "batch": 128,
            "lr": 0.05,
            "dim": 20,             # reduced these from 200
            "negative": 1,         # reduced from 5
            "opt": 'adagrad',
            "l2_reg": 0.001,
            "gradclip": 5,
            'filtered': True}
# margin
# cp_ratio
# metric
# nbest
# batch
# save_step

    algorithm.learn_embeddings(parameters)

    # algorithm.save_model(input_file_path + "analogy.mod")

    # algorithm.load_model(input_file_path + "analogy.mod")

    evaluate_file_names = {"test": input_file_path + "test.txt",
                           "whole": input_file_path + "whole.txt"}
                           # "relations": input_file_path + "relation2id.txt",
                           # "entities": input_file_path + "entity2id.txt"}

    algorithm.evaluate(evaluate_file_names)

    #    ents = ['/m/08mbj32', '/m/08mbj5d', '/m/08mg_b']
    test_subs = ['/m/07z1m', '/m/03gqgt3', '/m/01c9f2']
    test_rels = ['/location/statistical_region/religions./location/religion_percentage/religion',
                 '/military/military_conflict/combatants./military/military_combatant_group/combatants',
                 '/award/award_category/winners./award/award_honor/ceremony']
    tst_obs = ['/m/0631_', '/m/0d060g', '/m/01bx35']

    algorithm.retrieve_entity_embeddings(test_subs)

    sm = algorithm.retrieve_scoring_matrix(test_subs, test_rels)
    print(sm)

    # return output_file_path


if __name__ == '__main__':
    main(INPUT_FILE_DIRECTORY)
