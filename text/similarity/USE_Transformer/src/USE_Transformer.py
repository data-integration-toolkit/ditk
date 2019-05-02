import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import abc
from scipy import stats
from text.similarity.text_similarity import TextSemanticSimilarity
from decimal import Decimal

class USE_Transformer_Similarity(TextSemanticSimilarity):

    module_path = ''
    sentences_1 = []
    sentences_2 = []
    annotated_score = []

    def read_dataset(self, fileNames, *args, **kwargs):
        file_type = os.path.splitext(fileNames)[-1][1:]
        if file_type == 'csv':
            print("reading input csv file ")
            df = pd.read_csv(fileNames)
            self.sentences_1 = list(df['sentence_A'])
            self.sentences_2 = list(df['sentence_B'])
            self.annotated_score = list(df['relatedness_score'])
            del df
        elif file_type == 'txt':
            print("reading input txt file")
            df = pd.read_csv(fileNames,names=['sentence_A','sentence_B'])
            self.sentences_1 = list(df['sentence_A'])
            self.sentences_2 = list(df['sentence_B'])
            del df
        else:
            print("sorry,the DataSet format should be csv or txt")
            exit(-1)

    def train(self, *args, **kwargs):
        pass
    def save_model(self, *args, **kwargs):
        pass
    def load_model(self,arg_path):
        print("loading model...")
        self.module_path = arg_path
    def generate_embeddings(self, input_list, *args, **kwargs):
        pass

    def predict(self, data_X, data_Y, *args, **kwargs):
        tf.logging.set_verbosity(tf.logging.ERROR)
        # loading module
        encoder = hub.Module(self.module_path,trainable=True)
        # set tf.placeholder variables
        sentences1 = tf.placeholder(tf.string, shape=(None))
        sentences2 = tf.placeholder(tf.string, shape=(None))
        # generate embeddings and norm it
        sts_embed_1 = tf.nn.l2_normalize(encoder(sentences1), axis=1)
        sts_embed_2 = tf.nn.l2_normalize(encoder(sentences2), axis=1)
        # compute similarity of two embeddings
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_embed_1, sts_embed_2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        ag_sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
        print("encoding sentences and calculating sim scores, may takes while... ")
        # run tf session and transfer data to tf.placeholder variables by feed_dict
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sim_scores = session.run(cosine_similarities, feed_dict={sentences1: data_X, sentences2: data_Y})
        print("finished")
        return sim_scores.tolist()

    def evaluate(self, actual_values, predicted_values, *args, **kwargs):
        pearson_correlation = stats.pearsonr(predicted_values, actual_values)
        r_score = Decimal(pearson_correlation[0]).quantize(Decimal('0.00'))
        print('Pearson correlation coefficient = {0}'.format(
            r_score))
