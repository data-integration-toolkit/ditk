from text.similarity.text_similarity import TextSemanticSimilarity
import pandas as pd
import sys
from decimal import Decimal
from text.similarity.Siamese_LSTM.src.Siamese_LSTM import Siamese_LSTM
import pickle
import os
import scipy.stats as meas

class Siamese_LSTM_Similarity(TextSemanticSimilarity):

    def __init__(self):
        self.sentences_1 = []
        self.sentences_2 = []
        self.annotated_score = []
        self.model = None

    def read_dataset(self, file_name, *args, **kwargs):
        file_type = os.path.splitext(file_name)[-1][1:]
        if file_type == 'csv':
            print("reading input csv file ")
            df = pd.read_csv(file_name)
            self.sentences_1 = list(df['sentence_A'])
            self.sentences_2 = list(df['sentence_B'])
            self.annotated_score = list(df['relatedness_score'])
            del df
        elif file_type == 'txt':
            print("reading input txt file")
            df = pd.read_csv(file_name, names=['sentence_A', 'sentence_B'])
            self.sentences_1 = list(df['sentence_A'])
            self.sentences_2 = list(df['sentence_B'])
            del df
        else:
            print("sorry,the DataSet format should be csv or txt")
            exit(-1)

    def train(self, train_Data, max_epochs, *args, **kwargs):
        self.model = Siamese_LSTM(training=True)
        self.model.train_lstm(train_Data,max_epochs)

    def save_model(self, directory):
        path = directory + '/new.p'
        sys.setrecursionlimit(5000)  # avoid limit-exceeded when pickling
        pickle.dump(self.model, open(path, "wb"))

    def load_model(self, file):
        self.model = Siamese_LSTM(nam=file,load=True)
    def generate_embeddings(self, input_list, *args, **kwargs):
        # included in predict
        pass

    def predict(self, data_X, data_Y, *args, **kwargs):
        sim_score = self.model.predict_similarity( data_X, data_Y)
        return sim_score

    def evaluate(self, actual_values, predicted_values, *args, **kwargs):
        pearson_correlation = meas.pearsonr(predicted_values, actual_values)
        r_score = Decimal(pearson_correlation[0]).quantize(Decimal('0.00'))
        print('Pearson correlation coefficient = {0}'.format(
            r_score))

