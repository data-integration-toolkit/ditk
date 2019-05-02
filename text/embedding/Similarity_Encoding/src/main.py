import os
import sys
import numpy as np

from text_embedding_similarity import SimilarityEncoding
#from fit_predict_categorical_encoding import fit_predict_categorical_encoding

def main(train_input, train_output, predict_input, predict_output, embedding_dimension, similarity_input1, similarity_input2):
    # initialize Similarity Embedding
    te = SimilarityEncoding(embedding_dimension)

    # read training data
    te.read_Dataset(train_input)

    # train on training data
    output = te.train()
    unWords = np.unique(te.input_data)

    # output training data to a file(train_output)
    index = 0
    file = open(train_output,"w")
    for x in output.tolist():
        file.write(str(unWords[index]) + ' ' )
        file.write(str(x))
        file.write("\n")
        index += 1
    file.close()

    # predict embeddings:
    output = te.predict_embedding(predict_input)
    unWords = np.unique(te.predict_data)
    output = output.tolist()
    index = 0
    file = open(predict_output,"w")
    for x in output:
        file.write(str(unWords[index]) + ' ' )
        file.write(str(x))
        file.write("\n")
        index += 1
    file.close()

    # predict similarity between two strings
    # similarity_input1, similarity_input2
    sim = te.predict_similarity(similarity_input1, similarity_input2)
    print("similarity between \'" + similarity_input1 + "\' and \'" + similarity_input2 + "\' is: ")
    print(sim)


if __name__ == '__main__':
    train_input = '../data/sample_data/input.txt'
    train_output = '../data/sample_data/output.txt'
    predict_input = '../data/sample_data/predict.txt'
    predict_output = '../data/sample_data/predict_output.txt'
    embedding_dimension = 300
    similarity_input1 = 'Viterbi'
    similarity_input2 = 'Engineering'

    main(train_input, train_output, predict_input, predict_output, embedding_dimension, similarity_input1, similarity_input2)
