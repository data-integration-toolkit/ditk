from graph.embedding.analogy.analogy import ANALOGY

INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\FB15k"


# def main(input_file_path):
def main(input_file_path):
    # output_file_path = ""

    print("Main Script")

    algorithm = ANALOGY()

    file_names = {"train.txt", "valid.txt", "whole.txt"}
    algorithm.read_dataset(file_names)

    data = {}
    algorithm.learn_embeddings(data)

    # return output_file_path


if __name__ == '__main__':
    main(INPUT_FILE_DIRECTORY)
