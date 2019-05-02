import text_embedding
import os
import sys

import pandas as pd
import argparse
from multiprocessing import Process
from multiprocessing.queues import Empty

from ftodtf.cli import SETTINGS
from tqdm import tqdm

import ftodtf.model
import ftodtf.training
import ftodtf.input
import ftodtf.inference
import ftodtf.export
from ftodtf.settings import FasttextSettings
import ftodtf.model as model
import tensorflow as tf
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
#import tensorflow.python3.util.deprecation as deprecation

# class FastText(text_embedding.TextEmbedding):
class FastText(text_embedding.TextEmbedding):
    """
	A class which implements word vectors for generating text embeddings.
	Inherits from the parent class TextEmbedding
	"""

    def __init__(self, input_corpus_path, modelpath):
        self.settings = FasttextSettings()
        self.settings.log_dir = modelpath

    def read_Dataset(self,filePath, dataset_name):
        if dataset_name == 'sick':
            train_dir, test_dir = self.readSick(filePath)
        if dataset_name == 'semeval':
            train_dir, test_dir = self.readSemEval(filePath)
        if dataset_name == 'semeval2014':
            train_dir, test_dir = self.readSemEval2014(filePath)
        return train_dir,test_dir

    def preprocess(self):
        ipp = ftodtf.input.InputProcessor(self.settings)
        ipp.preprocess()
        print ('The input has been pre-processed.')
        try:
            print ('The input has been written to batches - ready to train.')
            ftodtf.input.write_batches_to_file(ipp.batches(),
                                               self.settings.batches_file,
                                               self.settings.num_batch_files)
        except Warning as w:
            # write_batches_to_file will raise a warning if there is not enough input-data
            print(w)
            sys.exit(1)

    def readSick(self, filePath):
        df = pd.read_excel(filePath)
        train_data = ""
        test_data_list = []
        for index, row in df.iterrows():
            if row['SemEval_set'] == 'TRAIN':
                train_data = train_data + row['sentence_A'] + '.\n' + row['sentence_B'] + '.\n'
        #            print(train_data)
            if row['SemEval_set'] == 'TEST':
                test_data_row_list = []
                test_data_row_list.append(row['sentence_A'])
                test_data_row_list.append(row['sentence_B'])
                test_data_row_list.append(row['relatedness_score'])
                test_data_list.append(test_data_row_list)

        current_dir = os.getcwd()
        #print('Current dir is', current_dir)
        train_dir = os.path.join(current_dir, 'train_sick.txt')
        test_dir = os.path.join(current_dir,'test_sick.txt')
        train_fp = open(train_dir, 'w')
        test_fp = open(test_dir,'w')
        train_fp.write(train_data)
        test_fp.write(str(test_data_list))
        self.settings.corpus_path = train_dir
        print ('The dataset has been split into train and test based on labels in input dataset')
        #print(self.settings.corpus_path)

        # Validating the folder details
        try:
            self.settings.validate_preprocess()
        except Exception as e:
            print(": ".join(["ERROR", e.__str__()]))
            sys.exit(1)
        
        self.preprocess()
        return train_dir, test_dir

    def readSemEval(self,filePath):
        
        #df=pd.read_csv(filePath,sep='\t',engine='python')
        df = pd.read_csv(filePath,error_bad_lines=False,sep='\t',usecols=range(7),engine='python')
        train_data=""
        test_data_list = []
        for index, row in df.iterrows():
            if 'test' in row[2]:
                test_data_row_list=[]
                test_data_row_list.append(row[5])
                test_data_row_list.append(row[6])
                test_data_row_list.append(row[4])
                test_data_list.append(test_data_row_list)
            else:
                train_data=train_data+' '+row[5]+'.\n'+row[6]+'.\n'
        current_dir = os.getcwd()
        #print('Current dir is', current_dir)
        train_dir = os.path.join(current_dir, 'train_semeval.txt')
        test_dir = os.path.join(current_dir,'test_semeval.txt')
        train_fp = open(train_dir, 'w')
        test_fp = open(test_dir,'w')
        train_fp.write(train_data)
        test_fp.write(str(test_data_list))
        self.settings.corpus_path = train_dir
        try:
            self.settings.validate_preprocess()
        except Exception as e:
            print(": ".join(["ERROR", e.__str__()]))
            sys.exit(1)
        self.preprocess()
        return train_dir, test_dir

    
    def readSemEval2014(self,filePath):
        df = pd.read_excel(filePath,columns = ["Sentence1", "Sentence2", "Similarity"])
        df.columns = ["Sentence1", "Sentence2", "Similarity"]
        train_data = ""
        test_data_list = []
        for index, row in df.iterrows():
            if index <= 2040:
                train_data = train_data + row['Sentence1'] + '.\n' + row['Sentence2'] + '.\n'
            elif index > 2040:
                test_data_row_list = []
                test_data_row_list.append(row['Sentence1'])
                test_data_row_list.append(row['Sentence2'])
                test_data_row_list.append(row['Similarity'])
                test_data_list.append(test_data_row_list)

        current_dir = os.getcwd()
        print('Current dir is', current_dir)
        train_dir = os.path.join(current_dir, 'train_semeval2014.txt')
        test_dir = os.path.join(current_dir,'test_semeval2014.txt')
        train_fp = open(train_dir, 'w')
        test_fp = open(test_dir,'w')
        train_fp.write(train_data)
        test_fp.write(str(test_data_list))
        self.settings.corpus_path = train_dir
        print(self.settings.corpus_path)

            # Validating the folder details
        try:
            self.settings.validate_preprocess()
        except Exception as e:
            print(": ".join(["ERROR", e.__str__()]))
            sys.exit(1)
        self.preprocess()
        #test_dir = '/home/khadutz95/FToDTF/train_sick.txt'
        return train_dir, test_dir
        
    def readEnwik8(self,filePath):
        self.settings.corpus_path = filePath
        print(self.settings.corpus_path)

        ipp = ftodtf.input.InputProcessor(self.settings)
        ipp.preprocess()
        try:
            print (self.settings.corpus_path)
            ftodtf.input.write_batches_to_file(ipp.batches(),
                                               self.settings.batches_file,
                                               self.settings.num_batch_files)
        except Warning as w:
            # write_batches_to_file will raise a warning if there is not enough input-data
            print(w)
            sys.exit(1)
        test_dir = '/home/khadutz95/FToDTF/train_sick.txt'
        return train_dir, test_dir

    def load_model(self,filePath):
        m = model.InferenceModel(self.settings)
        with tf.Session(graph=m.graph) as sess:
            m.load(self.settings.log_dir, sess)
            emb = sess.run(m.embeddings)
            ipp = ftodtf.input.InputProcessor(self.settings)
            dict_mapping = {}
            dict_mapping = ipp.preprocess()
        return emb, dict_mapping
    
    def save_model(self,filePath):
        np.savetxt(filePath,sess.run(self.embeddings), fmt="%s", delimiter = ",")

    def train(self,filepath):
        try:
            self.settings.validate_train()
        except Exception as e:
            print (e)
            print('Error in validation training settings')
            sys.exit(1)
        print('Training starting..')
        #ftodtf.training.train(self.settings)
        m = model.InferenceModel(self.settings)
        with tf.Session(graph=m.graph) as sess:
            print ('Loading a pre-trained model from', self.settings.log_dir)
            m.load(self.settings.log_dir, sess)
            emb = sess.run(m.embeddings)
            ipp = ftodtf.input.InputProcessor(self.settings)
            dict_mapping = {}
            dict_mapping = ipp.preprocess()
        return emb, dict_mapping

    def sent_vectorizer(self, sent,model,dict_mapping):
        
        sent_vec = np.zeros(200)
        numw = 0
        for w in sent:
            # print (w)
            # print (self.predict_embedding(w),w)
            #print (model[0][dict_mapping[w]])
            try:
                # print (type(self.predict_embedding(w))
                sent_vec = np.add(sent_vec,model[0][dict_mapping[w]])
                numw+=1
            except:
                pass
                #print ('Averaging word vectors failed')
        return sent_vec / np.sqrt(sent_vec.dot(sent_vec))

    def predict_embedding(self,input,model,dict_mapping):
        # print('In predict')
        if len(input) == 1:
            try:
                predicted_embedding = model[0][dict_mapping[input[0]]]
                return predicted_embedding
            except:
                print ("Cannot predict embedding for Out of Vocabulary words")
                return 'None'
        else:
            return self.sent_vectorizer(input,model,dict_mapping)
            
    def predict_similarity(self,input1,input2,model,dict_map):
        input1 = input1.split()
        input2 = input2.split()
        try:
            sent1_emb = self.predict_embedding(input1,model,dict_map)
            sent2_emb = self.predict_embedding(input2,model,dict_map)
            sent1_emb = sent1_emb.reshape(1, -1)
            sent2_emb = sent2_emb.reshape(1,-1)
            similarity = cosine_similarity(sent1_emb, sent2_emb, dense_output=True)
            return similarity
        except:
            return 'None'
        '''input_list = []
        input_list.append(input1)
        input_list.append(input2)
        ftodtf.inference.compute_similarities(input_list,self.settings)
        ipp = ftodtf.input.InputProcessor(self.settings)
        dict_mapping = {}
        dict_mapping = ipp.preprocess()'''
    
    def new_pred_similarity(self,input1):
        sim_score = ftodtf.inference.compute_similarities(input1, self.settings)           
        return sim_score

    def evaluate(self,model,filePath,evaluation_type,dict_map):
        if evaluation_type=='sick':
            train_fp = open(filePath,'r')
            test_data = train_fp.read()
            test_data = ast.literal_eval(test_data)
            predicted_sim_list=[]
            actual_sim_list=[]
            #print (len(test_data))
            #print(test_data)
            for each_list in test_data:
                sentence = []
                sentence1=each_list[0]
                #sentence1 = sentence1.split()
                sentence2=each_list[1]
                #sentence2 = sentence2.split()
                sentence.append(sentence1)
                sentence.append(sentence2)
                sim_score=self.predict_similarity(sentence1,sentence2,model,dict_map)                  
                predicted_sim_list.append(sim_score[0][0])
                actual_sim_list.append(each_list[2]/5.0)
            eval_score, p = pearsonr(actual_sim_list,predicted_sim_list)
            mean_sq_err = mean_squared_error(actual_sim_list,predicted_sim_list)
            spearman, metrics = spearmanr(actual_sim_list,predicted_sim_list)
            return eval_score,mean_sq_err,spearman

'''def main():
    # Make an object of Fasttext Settings
    
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    #deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.logging.set_verbosity(tf.logging.ERROR)
    dirpath = os.getcwd()
    datapath = dirpath + "/data/sick.xls"
    modelpath = dirpath + "/models/log_sick/log/"
    print (modelpath)

    #For SICK    
    fasttext_obj = FastText(input_corpus_path=datapath, modelpath = modelpath)

    # Call read_dataset of corresponding method
    train_data_path,test_data_path = fasttext_obj.read_dataset(datapath,'sick')

    # Call train
    model, dict_map = fasttext_obj.train(train_data_path)

    # Call Predict
    pred_embedding = fasttext_obj.predict_embedding(['things','kettle'],model, dict_map)
    print ('Predicted embedding is',pred_embedding)
    
    similarity = fasttext_obj.predict_similarity(['cheetah'],['hamster'],model, dict_map)
    print (similarity[0][0])

    # Call evaluate
    eval_score, mean_score,spearman_score = fasttext_obj.evaluate(model,test_data_path,'sick',dict_map)
    print (eval_score, mean_score,spearman_score)

if __name__ == '__main__':
    main()'''

