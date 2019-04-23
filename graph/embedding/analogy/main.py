import os
from graph.embedding.analogy.analogy import ANALOGY

INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\FB15k\\"

# input_file_path = ""
# output_file_path = main(input_file_path)


def main(input_file_path):
    # output_file_path = ""

    print("Main Script")

    algorithm = ANALOGY()

    train_file_names = {"train": input_file_path + "train.txt",
                        "valid": input_file_path + "valid.txt",
                        "whole": input_file_path + "whole.txt",
                        "relations": input_file_path + "relation2id.txt",
                        "entities": input_file_path + "entity2id.txt"}

    algorithm.read_dataset(train_file_names)

    parameters = {"mode": 'single',
            "epoch": 3,
            "batch": 128,
            "lr": 0.05,
            "dim": 50,             # reduced these from 200
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

    algorithm.save_model(input_file_path + "\\analogy.md")

    evaluate_file_names = {"test": input_file_path + "test.txt",
                           "whole": input_file_path + "whole.txt",
                           "relations": input_file_path + "relation2id.txt",
                           "entities": input_file_path + "entity2id.txt"}

    algorithm.evaluate(evaluate_file_names)

    # return output_file_path


if __name__ == '__main__':
    main(INPUT_FILE_DIRECTORY)
