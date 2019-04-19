import os
from graph.embedding.analogy.analogy import ANALOGY

INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\FB15k\\"


def main(input_file_path):
    # output_file_path = ""

    print("Main Script")

    algorithm = ANALOGY()

    file_names = {"train": input_file_path + "train.txt",
                  "valid": input_file_path + "valid.txt",
                  "whole": input_file_path + "whole.txt",
                  "relations": input_file_path + "relation2id.txt",
                  "entities": input_file_path + "entity2id.txt"}

    '''
    file_names = {"train": "D:\\USC\\CS548\\projectcode\\dat\\FB15k\\freebase_mtr100_mte100-train.txt",
                  "valid": "D:\\USC\\CS548\\projectcode\\dat\\FB15k\\freebase_mtr100_mte100-valid.txt",
                  "whole": "D:\\USC\\CS548\\projectcode\\dat\\FB15k\\freebase_mtr100_mte100-valid.txt",
                  "relations": "D:\\USC\\CS548\\projectcode\\dat\\FB15k\\train.rellist",
                  "entities": "D:\\USC\\CS548\\projectcode\\dat\\FB15k\\train.entlist"}
    '''

    algorithm.read_dataset(file_names)

    data = {"mode": 'single',
            "epoch": 500,
            "batch": 128,
            "lr": 0.05,
            "dim": 200,
            "negative": 5,
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

    algorithm.learn_embeddings(data)

    # return output_file_path


if __name__ == '__main__':
    main(INPUT_FILE_DIRECTORY)
