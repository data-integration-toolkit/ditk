import sys
import os
import tensorflow as tf
from graph_embedding_transE import MyTransE

def main():
    input_folder = './data/YAGO/'
    output_folder = './output_'
    dimension = 300
    marginal_value = 1.0
    batch_size = 4800
    max_epoch = 3

    graph_embedding = MyTransE()

    train, validation, test = graph_embedding.read_dataset(input_folder)

    embedding_vector, n_entity, n_relation = graph_embedding.learn_embeddings(dimension, marginal_value, batch_size, max_epoch)

    evaluations = graph_embedding.evaluate()

    print('-----Average-----')
    print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(evaluations['MRR'],evaluations['Hits']))

    graph_embedding.save_model(output_folder)

if __name__ == '__main__':
    main()
