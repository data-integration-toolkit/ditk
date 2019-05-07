import os
from graph.embedding.analogy.analogy import ANALOGY

INPUT_FILE_DIR = "D:\\USC\\CS548\\ditk\\graph\\embedding\\analogy\\test\\"

# FB15k
INPUT_FILE_NAME = "G9_embedding_test_input_FB15k.txt"
# WN18
# INPUT_FILE_NAME = "G9_embedding_test_input_WN18.txt"


def main(input_file_path):

    print("Running main - input file: " + input_file_path)

    file_names = {}
    parameters = {}
    input_subjects = []
    input_relations = []
    input_objects = []
    with open(input_file_path, "r") as f:
        # read in the input file assumptions for training
        tokens = f.readline().split(',')
        if len(tokens) != 10:
            print("Error reading first line in inputfile, check if correct number of items")
            print(" First line should contain:)")
            print(" <path_to_input_files>,<train_file>,<validate_file>,<test_file>, <whole_ext_file>,"
                  "<epochs>,<dimensions>,<output_file_path>")
            return None

        # ################################################################ fix this to handle os.path ????
        input_data_path = tokens[0].strip('"').strip()
        file_names['train'] = input_data_path + tokens[1].strip('"').strip()
        if len(tokens[2].strip('"').strip()) != 0:
            file_names['valid'] = input_data_path + tokens[2].strip('"').strip()
        if len(tokens[3].strip('"').strip()) != 0:
            file_names['whole'] = input_data_path + tokens[3].strip('"').strip()
        file_names['test'] = input_data_path + tokens[4].strip('"').strip()

        file_names['relations'] = input_data_path + tokens[5].strip('"').strip()
        file_names['entities'] = input_data_path + tokens[6].strip('"').strip()
        parameters['epoch'] = int(tokens[7])
        parameters['dim'] = int(tokens[8])
        file_names['output'] = tokens[9].strip('"').strip()

        # read in test triplets
        for line in f.readlines():
            s, r, o = line.strip("\n").split("\t")
            input_subjects.append(s)
            input_relations.append(r)
            input_objects.append(o)

    algorithm = ANALOGY()

    algorithm.read_dataset(file_names)

    algorithm.learn_embeddings(parameters)

    # retrieve embeddings
    sub_re_em, sub_im_em, sub_em = algorithm.retrieve_entity_embeddings(input_subjects)
    rel_re_em, rel_im_em, rel_em = algorithm.retrieve_relations_embeddings(input_relations)
    obj_re_em, obj_im_em, obj_em = algorithm.retrieve_entity_embeddings(input_objects)

    # output the vectors to a file
    with open(file_names['output'], "w") as f:
        for i in range(len(input_subjects)):
            f.write("{0}: {1}\n".format(input_subjects[i], sub_em[i]))
            f.write("{0}: {1}\n".format(input_relations[i], rel_em[i]))
            f.write("{0}: {1}\n".format(input_objects[i], obj_em[i]))
            f.write("\n\n")

    return file_names['output']


if __name__ == '__main__':
    output_file_path = main(INPUT_FILE_DIR + INPUT_FILE_NAME)
    if output_file_path is not None:
        print("Embeddings for input file outputted to:\n   " + output_file_path)
