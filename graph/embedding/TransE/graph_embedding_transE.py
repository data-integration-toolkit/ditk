"""
This is my method class
"""
import abc
from graph_embedding import Graph_Embedding
import numpy as np

# Import original transE source codes
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/src")
from main import main

from dataset import KnowledgeGraph
from model import TransE

import tensorflow as tf
import argparse
import numpy as np


class MyTransE(Graph_Embedding):
    def __init__(self):
        graph_original_entity = []
        graph_embedded_entity = None
        graph_embedded_relations = None
        dataset_name = ""
        self.kg = None
        self.model = None
        pass

    #@abc.abstractmethod
    def read_dataset(self, fileNames, options = {}):
        # read the data from fileNames
        # parse the data into a format this method can use
        """
        read and store the data_foler_path
        """
        self.kg = KnowledgeGraph(fileNames)
        return self.kg.training_triples, self.kg.validation_triples, self.kg.test_triples
        #self.dataset_name = fileNames
        #pass

    #@abc.abstractmethod
    def learn_embeddings(self, dim, margin, batch, max_epoch):
        #if self.dataset_name == "WN18":
        #    self.graph_embedded_entity, self.graph_embedded_relations = main(data_dir='./data/WN18/')
        # learn the embedding of the relations
        # store the embedding to TransE.graph_embeded
        #pass

        tf.reset_default_graph()
        parser = argparse.ArgumentParser(description='TransE')
        """
        #parser.add_argument('--data_dir', type=str, default='./data/None/')
        parser.add_argument('--embedding_dim', type=int, default=300)
        parser.add_argument('--margin_value', type=float, default=1.0)
        parser.add_argument('--score_func', type=str, default='L1')
        parser.add_argument('--batch_size', type=int, default=4800)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--n_generator', type=int, default=24)
        parser.add_argument('--n_rank_calculator', type=int, default=24)
        parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
        parser.add_argument('--summary_dir', type=str, default='../summary/')
        parser.add_argument('--max_epoch', type=int, default=3)
        ##parser.add_argument('--max_epoch', type=int, default=3)
        parser.add_argument('--eval_freq', type=int, default=10)
        args = parser.parse_args()
        print(args)
        """
        kg = self.kg
        #kge_model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
        #                   score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate,
        #                   n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator)

        kge_model = TransE(kg=kg, embedding_dim=dim, margin_value=margin,
                           score_func='L1', batch_size=batch, learning_rate=0.001,
                           n_generator=24, n_rank_calculator=24)

        gpu_config = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_config)
        with tf.Session(config=sess_config) as sess:
            #tf.reset_default_graph()
            print('-----Initializing tf graph-----')
            tf.global_variables_initializer().run()
            print('-----Initialization accomplished-----')
            kge_model.check_norm(session=sess)
            summary_writer = tf.summary.FileWriter(logdir='../summary/', graph=sess.graph)
            for epoch in range(max_epoch):
                print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
                kge_model.launch_training(session=sess, summary_writer=summary_writer)
                #============================================================
                #params = [kge_model.entity_embedding]
                #param = []
                #for each in params:
                #    param.append(np.array(each.eval()))
                #param = np.array(param)
                #np.save('./a.npy', param)
                #=============================================================
                #if (epoch + 1) % args.eval_freq == 0:
                #    kge_model.launch_evaluation(session=sess)
    	    #============================================================
            params = [kge_model.entity_embedding, kge_model.relation_embedding]
            param = []
            for each in params:
                param.append(np.array(each.eval()))
            param = np.array(param)
            #    np.save('./a.npy', param)

            self.graph_embedded_entity, self.graph_embedded_relations = param
            self.model = kge_model

        return np.concatenate((self.graph_embedded_entity, self.graph_embedded_relations), axis=0), self.kg.n_entity, self.kg.n_relation

    #@abc.abstractmethod
    def evaluate(self):
        # Use TransE.graph_embeded for evaluation
        # Evaluation:   predicts tail entities
        #               Cosine similarity
        gpu_config = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_config)
        with tf.Session(config=sess_config) as sess:
            tf.initialize_all_variables().run()
            head_meanrank_filter, head_hits10_filter, tail_meanrank_filter, tail_hits10_filter = self.model.launch_evaluation(session=sess)

        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                                 (head_hits10_filter + tail_hits10_filter) / 2))
        eval_dict = {}
        #eval_dict['f1'] =
        eval_dict['MRR'] = (head_meanrank_filter + tail_meanrank_filter) / 2
        eval_dict['Hits'] = (head_hits10_filter + tail_hits10_filter) / 2
        return eval_dict

    def save_model(self,output_path):
        #np.save(output_path + 'embedded_entity.npy', self.graph_embedded_entity)
        #np.save(output_path + 'embedded_relations.npy', self.graph_embedded_relations)
        #self.entity_dict


        file = open(output_path + 'entity_dict.txt',"w")

        i = 0
        for key,value in self.kg.entity_dict.items():
            file.write(str(key) + ' ')
            for num in self.graph_embedded_entity[i][:-1]:
                file.write(str(num) + ',')
            file.write(str(self.graph_embedded_entity[i][-1]))
            file.write("\n")
            i += 1
        file.close()


        np.savetxt(output_path + 'embedded_entity.out', self.graph_embedded_entity, delimiter=',')
        np.savetxt(output_path + 'embedded_relations.out', self.graph_embedded_relations, delimiter=',')

    def load_model(self, file):
        pass
