import abc
#from fns_categorical_encoding import categorical_encoding
import numpy as np
import pandas as pd
import math
import sys,os
# Import original transE source codes

sys.path.append(os.path.join(os.path.abspath("../"), "src"))
from fns_categorical_encoding import categorical_encoding

class SimilarityEncoding():
    def __init__(self, dim):
        self.input_filename = None
        self.input_data = []
        self.predict_data = []
        self.embeddings = None
        self.dimension = dim

    def read_Dataset(self, fileName):
        self.input_filename = fileName

        data = []
        with open(fileName) as fp:
           line = fp.readline()
           while line:
               data.append(line)
               line = fp.readline()
        fp.close()

        self.input_data = np.asarray(data)

    def read_Predict_Dataset(self, fileName):
        self.input_filename = fileName

        data = []
        with open(fileName) as fp:
           line = fp.readline()
           while line:
               data.append(line)
               line = fp.readline()
        fp.close()

        self.predict_data = np.asarray(data)

    def train(self):

        #a = ['apple likes','apple likes','apple likes','apple likes']
        #b = ['apple','banana','cat','cats', 'cats']
        #A = np.asarray(a)
        #B = np.asarray(a)
        encoder = categorical_encoding(self.input_data, self.input_data, None,'levenshtein-ratio_SimilarityEncoder', None, None)
        if len(encoder) < self.dimension:
            repeat = math.ceil(self.dimension * 1.0 /len(encoder))
            encoder = np.repeat(encoder, repeat, axis=1)
        self.embeddings = encoder[:self.dimension]
        return encoder[:self.dimension]

    def predict_embedding(self, input):
        self.read_Predict_Dataset(input)
        encoder = categorical_encoding(self.predict_data,self.input_data, None,'levenshtein-ratio_SimilarityEncoder', None, None)

        #print(self.dimension)
        #return encoder

        if len(encoder) < self.dimension:
            repeat = math.ceil(self.dimension * 1.0 /len(encoder[0]))
            encoder = np.repeat(encoder, repeat, axis=1)

        n,m = encoder.shape
        if m > self.dimension:
            encoder.resize((n, m-1))

        return encoder

    def predict_similarity(self, input1, input2):
        #input1 = string
        #input2 = string
        A = np.asarray([input1])
        B = np.asarray([input2])

        encoder1 = categorical_encoding(A, self.input_data, None,'levenshtein-ratio_SimilarityEncoder', None, None)

        if len(encoder1[0]) < self.dimension:
            repeat = math.ceil(self.dimension * 1.0 /len(encoder1[0]))
            encoder1 = np.repeat(encoder1, repeat, axis=1)

        n,m = encoder1.shape
        if m > self.dimension:
            encoder1.resize((n, m-1))

        encoder2 = categorical_encoding(B, self.input_data, None,'levenshtein-ratio_SimilarityEncoder', None, None)
        if len(encoder2[0]) < self.dimension:
            repeat = math.ceil(self.dimension * 1.0 /len(encoder2[0]))
            encoder2 = np.repeat(encoder2, repeat, axis=1)

        n,m = encoder2.shape
        if m > self.dimension:
            encoder2.resize((n, m-1))

        #print(encoder1)
        #print(encoder2)
        return np.corrcoef(encoder1, encoder2)[0,1]

    def evaluate(self, data_name, filename):
        # filename = sentence1, sentence2, sim_rate
        # predict(sentence1, sentence2, predicted_rate)
        # return sum(abs(sim_rate-predicted_rate))/n
        ret = 0
        if (data_name == "Semi2017"):
            my_data = pd.read_csv(filename, delimiter=',', header=None)
            my_data = my_data.loc[:,4:6]
            my_data.columns = ["sim_rate","S1","S2"]
            my_data["sim_rate"] = my_data.apply(lambda row: row["sim_rate"]/5,axis=1)
            my_data["pred_rate"] = my_data.apply(lambda row: self.predict_similarity(str(row['S1']),str(row['S2'])), axis=1)
            ret = my_data.apply(lambda row: abs(row['sim_rate'] - row['pred_rate']), axis=1).sum()/my_data.shape[0]
        else:
            print("data not recongnize")

        return ret
    def save_model(self, file):
        pass

    def load_model(self, file):
        pass
