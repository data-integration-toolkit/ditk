from text.similarity.Siamese_LSTM.src.Siamese_LSTM_Similarity import Siamese_LSTM_Similarity
from text.similarity.Siamese_LSTM.src.sentences import *
import pickle

if __name__ =='__main__':
    Syn_aug= True
    instance = Siamese_LSTM_Similarity()
    print("Pre-training")
    #  load sts_benchmark dataset, and the traindata format is like [list1,list2,...] ,every list is a list [s1,s2,sim_score]
    train_data = pickle.load(open("../data/stsallrmf.p", "rb"),encoding='latin-1')
    instance.train(train_data, 66)
    print("Pre-training done")
    if Syn_aug == True:
        # semtrain.p  semEval2014 train dataset
        train_data = pickle.load(open("../data/semtrain.p", 'rb'),encoding='latin-1')
        train_data = expand(train_data)
        instance.train(train_data, 375)
    else:
        instance.train(train_data, 330)

    instance.save_model('../model')
    test = pickle.load(open("../data/semtest.p", 'rb'))
    print(instance.model.chkterr2(test))


