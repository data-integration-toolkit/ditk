from modules.input_data import ProcessInputData
from modules.datasets import MSRPDataset
from modules.configs import MSRP_FILE

from keras.models import Model, load_model

import pickle
import sys

TOKENIZER = 'tokenizers/tokenizer_SICK_2200.pickle'
MODEL_FILE = 'results/model_1510006867660.h5'


def predict():
    # loading
    with open(TOKENIZER, 'rb') as handle:
        tokenizer = pickle.load(handle)

    preprocess = ProcessInputData(tokenizer=tokenizer)
    # sentence_1 = 'the girl is buying potatoes'
    # sentence_2 = 'the girl bought potatoes yesterday'
    # sentence_2 = 'the girl will buy potatoes tomorrow'

    # sentence_1 = 'Peter builds a house'
    # sentence_2 = 'A house is built by Peter'

    sentence_11 = 'Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier'
    sentence_12 = 'With the scandal hanging over Stewart\'s company, revenue the first quarter of the year dropped 15 percent from the same period a year earlier.'

    sentence_21 = 'The DVD CCA then appealed to the state Supreme Court'
    sentence_22 = 'The DVD CCA appealed that decision to the U.S. Supreme Court'

    sentence_31 = 'But he added group performance would improve in the second half of the year and beyond.'
    sentence_32 = 'De Sole said in the results statement that group performance would increase in the second half of the year and beyond.'

    x1, x2 = preprocess.get_input_from_collection([sentence_11, sentence_21, sentence_31],
                                                  [sentence_12, sentence_22, sentence_32], 32)
    network = load_model(MODEL_FILE)
    a = network.predict([x1, x2])
    print a
    a = (a * 4) + 1
    print a

    df = MSRPDataset(MSRP_FILE).data_frame()
    print df.shape[0]

    para = df.loc[df['label'] == 1]
    print para.shape

    # predict msrp
    s1 = []
    s2 = []
    for index, row in df.iterrows():
        s1.append(row['s1'])
        s2.append(row['s2'])
    x1, x2 = preprocess.get_input_from_collection(s1, s2, 32)
    a = network.predict([x1, x2])
    print a

    print min(a)
    mean = sum(a) / float(len(a))
    print mean
    print (mean * 4) + 1



if __name__ == "__main__":
    predict()

