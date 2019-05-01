from graph.embedding.HolE import holographic_embedding
import numpy as np


def main(fileName):
    example = holographic_embedding.HolE_Embedding()

    example.input_file = fileName
    prefix = ''
    if fileName[0].rfind('/') >= 0:
        prefix = fileName[0][:fileName[0].rfind('/')] + '/'

    example.output_file = prefix + 'output.txt'

    example.train, example.valid, example.test, example.entities, example.relations = example.read_dataset()

    [embedding_vector, model, ev_test] = example.learn_embeddings()

    c1 = open(example.output_file, "w")

    for i in range(len(model.E)):
        c1.write(str(example.entities[i]) + " " + str(np.array(model.E[i])) + '\n')
    for i in range(len(model.R)):
        c1.write(str(example.relations[i]) + " " + str(np.array(model.R[i])) + '\n')
    c1.close()

    result = example.evaluate(fileName, model, ev_test)
    print("cosine similarity: " + str(result['cosine similarity']) + '\n')
    print("MR: " + str(result['MR']) + '\n')
    print("Hits@1: " + str(result['Hits']['1']) + '\n')
    print("Hits@3: " + str(result['Hits']['3']) + '\n')
    print("Hits@10: " + str(result['Hits']['10']) + '\n')

    return example.output_file


if __name__ == '__main__':
    # example = HolE_Embedding()
    # example.fileName = ['data/yago/train_small.txt','data/yago/valid_small.txt','data/yago/test_small.txt']
    fileName = ['tests/test_input/train.txt', 'tests/test_input/valid.txt', 'tests/test_input/test.txt']
    output_file = main(fileName)
