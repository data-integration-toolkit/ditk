# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:49:52 2019

@author: Lavanya Malladi
"""

import text_embedding
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models import Word2Vec 
from scipy.stats import pearsonr
from gensim.utils import simple_preprocess    
from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  LinearSVC
import numpy
import numpy as np
import random
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import math
from scipy.spatial import distance
from random import sample
from nltk.corpus import stopwords
from scipy.stats import spearmanr

class PhraseVector:
    def __init__(self, phrase, model):
        self.vector = self.PhraseToVec(phrase, model)
    # <summary> Calculates similarity between two sets of vectors based on the averages of the sets.</summary>
    # <param>name = "vectorSet" description = "An array of arrays that needs to be condensed into a single array (vector). In this class, used to convert word vecs to phrases."</param>
    # <param>name = "ignore" description = "The vectors within the set that need to be ignored. If this is an empty list, nothing is ignored. In this class, this would be stop words."</param>
    # <returns> The condensed single vector that has the same dimensionality as the other vectors within the vecotSet.</returns>
    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
        if len(ignore) == 0: 
            return np.mean(vectorSet, axis = 0)
        else: 
            return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)

    def PhraseToVec(self, phrase, model):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector=model[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)

    # <summary> Calculates Cosine similarity between two phrase vectors.</summary>
    # <param> name = "otherPhraseVec" description = "The other vector relative to which similarity is to be calculated."</param>
    def CosineSimilarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity=0
        except:
            cosine_similarity=0        
        return cosine_similarity


class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)



semEval=False
sick=False
reviews=False
# sentences = word2vec.Text8Corpus('file://C:/Users/malla/Downloads/text8/text8')
# model = Word2Vec(sentences, iter=10, min_count=10, size=200, workers=4) 
train_semEval_data=""
train_sick_data=""
train_reviews_data=TaggedLineSentence({})



class Word2Vec_Util(text_embedding.TextEmbedding):
    
    
    '''
        Initializing the Benchmark Datasets to False 
    '''
    
    
    
    '''
        If no dataset is provided -- By default Text8 Corpus is utilized
        
    '''
    

   
    
    def __init__(self):
        self.sentences = word2vec.Text8Corpus('file://C:/Users/malla/Downloads/text8/text8')
        self.model = Word2Vec(self.sentences, iter=10, min_count=10, size=200, workers=4) 
        self.save_model('text8')
        pass

        
    
    def read_Dataset(self, name='text8', fileName='file://C:/Users/malla/Downloads/text8/'):
        '''
            User can mention the name and fileName ( path of the file ) that has to be fed to create the
            training model
            Input parameters :
                - name : file name
                - fileName : filepath
            Action : Call the respective method to read the dataset (sick, semEval, reviews)

        '''
        if name=='sick':
            global sick
            sick=True
            print(fileName)            
            return self.readSick(fileName+'sick_2014/')
        if name=='semEval':
            global semEval
            semEval=True            
            return self.readSemEval(fileName+'semEval_2017/sts-test.csv')
        if name=='reviews':
            global reviews
            reviews=True
            return self.readReviews(fileName)
            

    
    def readSemEval(self, input_file):
        '''
            Read Sem Eval Data into train and test list
            Returns :
                - Train_semEval_data : Training data
                - Test_sentence_list : Test data

        '''

        global train_semEval_data
        train_semEval_data=""
        test_sentence_list=[]
        df=pd.read_csv(input_file,error_bad_lines=False,sep='\t',usecols=range(7),engine='python')
        for index, row in df.iterrows():
            if 'test' in row[2]:
                test_list=[]
                test_list.append(row[5])
                test_list.append(row[6])
                test_list.append(row[4])
                test_sentence_list.append(test_list)
            else:
                train_semEval_data=train_semEval_data+' '+row[5]+'.\n'+row[6]+'.\n'
        return train_semEval_data, test_sentence_list
            
        
   
    
    
        
    def readReviews(self, fileName):
        '''
            Read reviews data 
            Returns :
                - Train_sentences : Training data
                - Test_sentences : Test data

        '''
        global reviews
        global train_reviews_data
        reviews=True
        train_source = {fileName+'train-neg.txt':'TRAIN_NEG', fileName+'train-pos.txt':'TRAIN_POS'}
        test_source = {fileName+'test-neg.txt':'TEST_NEG', fileName+'test-pos.txt':'TEST_POS'}
        train_sentences = TaggedLineSentence(train_source)
        test_sentences = TaggedLineSentence(test_source)
        train_reviews_data=train_sentences
        return train_reviews_data,test_sentences
        
    
    
    
    def readSick(self, input_file):
        ''' 

            Loads the SICK data set into training and test variables
            Returns:
                train_data (string): Corpus os text for training
                test_data_list (list): [sentence1, sentence2, actual_similarity_score]

        '''
        global train_data
        print(input_file+'/'+'SICK_train.xlsx')
        
        df_train = pd.read_excel(input_file+'/'+'SICK_train.xlsx') 
        df_test = pd.read_excel(input_file+'/'+'SICK_test_annotated.xlsx')
        train_data = ""
        test_data_list=[]
        data = []
        for index, row in df_train.iterrows():
                train_data=train_data+' '+row['sentence_A']+'.\n'+row['sentence_B']+'.\n'
        for index, row in df_test.iterrows():
                test_data_row_list=[]
                test_data_row_list.append(row['sentence_A'])
                test_data_row_list.append(row['sentence_B'])
                test_data_row_list.append(row['relatedness_score'])
                test_data_list.append(test_data_row_list)

        return(train_data, test_data_list)
        
        
    def save_model(self, file=""):
        '''
            Saves the model to a file
            Input parameters : file - location where the model should be saved
            Action : Save the model to the file

        '''
        self.model.save(file)
        
    def load_model(self, file=""):
        '''
            Loads the model from a file
            Input parametes : file - location where the model should be loaded from
            Returns : Model

        '''
        model=Word2Vec.load(file)
        return model
        
    def preprocess(self, input_data):
        '''
            Preprocess the data to a format usable by the train() method
            Sentence : list of words, Document - List of ( List of words )
            Input Parameters (string) : input_data ( training_data ) that has to be preprocessed
            Returns -  data( List of lists ) : Preprocessed data 

        '''

        f = input_data.replace("\n", " ") 
      
        data = [] 
        # iterate through each sentence in the file 
        for i in sent_tokenize(f): 
            temp = [] 
              
            # tokenize the sentence into words 
            for j in word_tokenize(i): 
                temp.append(j.lower()) 
          
            data.append(temp) 
        return data
    
    def train(self, train_data, embedding_size=200, type=""):
        '''
            Training on train_data, given the embedding size, type 
            Input parameters : 
                - train_data : data on which training should be performed
                - embedding_size : default = 200
                - type : default = CBOW ( Continuous Bag of words)
                         'sg' (SkipGram Model)

            Returns : Trained Model

        '''
        global semEval
        global reviews
        global sick
        global semEval
        global model
        global train_semEval_data
        global train_reviews_data
        flag = 0
        # Training on sick dataset
        if sick==True:
            flag = 1
            if type=="sg":
                self.model = gensim.models.Word2Vec(train_data, min_count = 1,  
                                      size = 200, window = 5, sg=1) 
            else:
                self.model = gensim.models.Word2Vec(train_data, min_count = 1,  
                                      size = 200, window = 5) 

        # Training on reviews dataset
        if reviews==True:
            flag = 1
            self.model = Doc2Vec(min_count=1, window=10, size=embedding_size, sample=1e-4, negative=5, workers=7,iter=50)
            self.model.build_vocab(train_reviews_data.to_array())

            self.model.train(train_reviews_data.sentences_perm(),total_examples=self.model.corpus_count,epochs=self.model.iter)
        
        # Training on semEval dataset
        if semEval== True:
            flag = 1
            if type=="sg":
    
                self.model = gensim.models.Word2Vec(train_semEval_data, min_count = 1,size = 200, window = 5, sg=1) 
            else:
                self.model = gensim.models.Word2Vec(train_semEval_data, min_count = 1,size = 200, window = 5) 

        else :
            if flag == 0: 
                self.model = gensim.models.Word2Vec(train_data, min_count = 1,  size = 200, window = 5)

        self.save_model("word2vec")

        return self.model
    
        
    def predict_similarity(self, model, sentence1, sentence2):
        '''
            Returns the similarity score between 2 sentences
            Input parameters    : 2 sentences
            Returns             : Similarity Score [0-1]

        '''

        return self.predict_similarity_util(model, sentence1, sentence2)
        
    
    def predict_similarity_util(self, model, sentence1, sentence2):
    
        '''
            Utility method for predicting similarity
            Input parameters   : 2 sentences and trained model
            Returns            : Similarity Score

        '''
        phraseVector1 = PhraseVector(sentence1,model)
        phraseVector2 = PhraseVector(sentence2,model)

        similarityScore  = phraseVector1.CosineSimilarity(phraseVector2.vector)
        return similarityScore
    
    
    def sent_vectorizer(self, sent):
        '''
            Converts the sentence to a vector
            Input parameters    :   Sentence
            Returns             :   Average Vector representing the sentence

        '''
        sent_vec = np.zeros(200)
        numw = 0
        for w in sent:
            try:
                sent_vec = np.add(sent_vec,self.model.wv[w])
                numw+=1
            except:
                pass
        return sent_vec / numw
        
    
    def predict_embedding(self, sentence):

        '''
            Predicts the embedding for the word ['word'] or sentence ['word1', 'word2', 'word3',...]
            Input Parameters :
                - Sentence ( List ) : List of words
                - Word ( List )     : List of length 1

            Returns          : Vector

        '''                
        # sentences = word2vec.Text8Corpus('file://C:/Users/malla/Downloads/text8/text8')
        # model = Word2Vec(sentences, iter=10, min_count=10, size=300, workers=4) 
        model=self.load_model('word2vec')
        if len(sentence)==1:
            # Word
            word=sentence[0]
            return self.model.wv[word]
        else:
            # Sentence
            return self.sent_vectorizer(sentence)
    
    
    def evaluate(self, model, filename="", evaluation_type=""):
        '''
            Evaluate the model ( method ) on the benchmark dataset to compute the Pearson Correlation Coefficient,
            Mean Squared Error and Spearman Correlation Coefficient
            Input parameters :
                - model             : model generated after training
                - filename          : filepath where the benchmark dataset is stored 
                - evaluation_type   : ['sick','reviews','semEval'] -- Benchmark Datasets

            Returns          : Correlation Coefficient Values

        '''
        global train_sick_data
        global train_semEval_data

        # SICK Dataset
        if evaluation_type=='sick':
            train_sick_data,test_data_list=self.read_Dataset("sick", filename) 
            # read_Dataset(name, fileName) -- fileName : Mention the path of the sick_2014 folder
            train_sick_data=self.preprocess(train_sick_data)
            return self.evaluate_util(model,train_sick_data,test_data_list,evaluation_type='sick')

        # REVIEWS Dataset
        if evaluation_type=='reviews':
            train_reviews_data,test_reviews_data=self.read_Dataset("reviews", filename)
            # read_Dataset(name, fileName) -- fileName : Mention the path of the reviews folder folder
            return self.evaluate_util(model, train_reviews_data,test_reviews_data,evaluation_type)

        # SemEval 2017 Dataset
        if evaluation_type=='semEval':
            train_semEval_data,test_data_list=self.read_Dataset("semEval", filename)
            # read_Dataset(name, fileName) -- fileName : Mention the path of the reviews folder folder

            train_semEval_data=self.preprocess(train_semEval_data)
            model1=self.train(train_semEval_data)
            return self.evaluate_util(model,train_semEval_data,test_data_list,evaluation_type='sick')
            
            
                        
        
    def words_closer_than(self, w1, w2):
        '''
            Get all words that are closer to w1 than w2 is to w1.

                Parameters:    
                w1 (str) – Input word.
                w2 (str) – Input word.
                Returns:    
                List of words that are closer to w1 than w2 is to w1.
                
                Return type:    
                list (str)
                
        '''
        pass
        
        
    
    def evaluate_util(self, model, train_sentences,test_data_list,evaluation_type):
        '''
            Utility method for evaluation -- it consists of the entire flow for evaluation
            Input Parameters :
                - model                       : The trained model
                - train_sentences( string )   : sentences that have to be trained ( From the dataset )
                - test_data_list ( list)      : list of lists containing test_sentence_1, test_sentence_1, actual similarity value
                - evaluation_type             : ['sick', 'semEval', 'reviews'] -- depends on the benchmark dataset
            Returns          :
                - Pearson Correlation Coefficient
                - Mean Squared Error
                - Spearman Correlation Coefficient

        ''' 

        from sklearn.metrics import mean_squared_error
        from scipy.stats import spearmanr
        global train_reviews_data

        # Evaluation on SICK Dataset or SemEval Dataset -- Based on Semantic Similarity
        if evaluation_type=='sick':
            result={}
            predicted_similarity_score_list=[]
            actual_similarity_score_list=[]
            for each_list in test_data_list:
                sentence1=each_list[0]
                sentence2=each_list[1]
                sim_score=self.predict_similarity_util(model,sentence1,sentence2)                  
                predicted_similarity_score_list.append(sim_score)
                actual_similarity_score_list.append(each_list[2]/5.0)
            eval_score,p_val=pearsonr(predicted_similarity_score_list,actual_similarity_score_list)
            eval_score1,p_val=spearmanr(predicted_similarity_score_list,actual_similarity_score_list)
            result['PearsonR']=eval_score
            result['MSE']=mean_squared_error(actual_similarity_score_list, predicted_similarity_score_list)
            result['SpearmanR']=eval_score1
            return result


        # Evaluation on Reviews Dataset -- Based on Sentimental Analayis
        if evaluation_type=='reviews':
            train_arrays = numpy.zeros((25000, 200))
            train_labels = numpy.zeros(25000)
            for i in range(12500):
                prefix_train_pos = 'TRAIN_POS_' + str(i)
                prefix_train_neg = 'TRAIN_NEG_' + str(i)
                train_arrays[i] = model.docvecs[prefix_train_pos]
                train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
                train_labels[i] = 1
                train_labels[12500 + i] = 0
    
    
            test_arrays = numpy.zeros((25000, 200))
            test_labels = numpy.zeros(25000)
            
            
            for index, i in enumerate(test_data_list):
                # prefix_test_pos = 'TEST_POS_' + str(i)
                # prefix_test_neg = 'TEST_NEG_' + str(i)
                feature = model.infer_vector(i[0])
                if index <12500:
                    test_arrays[index] = feature
                    test_labels[index] = 0
                else:
                    test_arrays[index] = feature
                    test_labels[index] = 1
            
            classifier = LinearSVC()
            classifier.fit(train_arrays, train_labels)    
            result={}
            result['Accuracy']=classifier.score(test_arrays, test_labels)
            return result

    def verify_reviews(self,filename):
        '''
            Method to verify the evaluation of Reviews dataset -- it contains the entire flow, standalone method
            Returns :
                - Accuracy of the sentimental analysis

        '''
        global train_reviews_data,test_reviews_data
        global model

        train_reviews_data,test_reviews_data=self.read_Dataset("reviews",fileName= filename)
        model=self.train(train_reviews_data)
        evaluation_score=self.evaluate(model,evaluation_type='reviews')
        print(evaluation_score['Accuracy'])
        
    def verify_sick(self, filename):
        '''
            Method to verify the SICK Dataset -- it contains the entire flow, standalone method
            Returns :
                - Pearson Correlation Coefficient
                - Mean Squared Error
                - Spearman Correlation Coefficient

        '''
        global train_preprocessed_data
        global test_data_list
        global reviews
        global sick
        global model
        evaluation_result_dict  =self.evaluate(self.model,filename, evaluation_type='sick')
        print('CBOW :')
        print(evaluation_result_dict['PearsonR'])
        print(evaluation_result_dict['MSE'])
        print(evaluation_result_dict['SpearmanR'])


        
    def verify_semEval(self, filename):
        '''
            Method to verify the SemEval Dataset -- it contains the entire flow, standalone method
            Returns :
                - Pearson Correlation Coefficient
                - Mean Squared Error
                - Spearman Correlation Coefficient

        '''

        evaluation_result_dict=self.evaluate(self.model,filename, evaluation_type='semEval')


        print('CBOW :')
        print(evaluation_result_dict['PearsonR'])
        print(evaluation_result_dict['MSE'])
        print(evaluation_result_dict['SpearmanR'])


        # print('Skip-Gram')
        # print(evaluation_result_dict_sg['PearsonR'])
        # print(evaluation_result_dict_sg['MSE'])
        # print(evaluation_result_dict_sg['SpearmanR'])


        
        

        


        
    def verify_predict_embedding(self, fileName):

        ''' 
            Method to verify the predict embedding method, standalone
            Input Parameters :
                - name          - file name
                - fileName      - file Path
            Returns          : Embedding for the word/sentence

        '''
        file_read = open(fileName+'input.txt', 'r')
        output_file_path = 'output.txt'
        file_write = open( output_file_path ,'w')
        word_list = file_read.read().split('\n')

        while True:
            for word in word_list:
                if word == None:
                    break
                if word == " ":
                    continue
                embedding = self.predict_embedding([word])
                file_write.write(word)
                file_write.write('\n')
                file_write.write(str(embedding))
                file_write.write('\n')

        return output_file_path



        print(self.predict_embedding(['man']))



        
'''
Sample main : Defining a Sample Workflow
def main():

    global train_sick_data
    word2vec_util=Word2Vec_Util()
    train_sick_data,test_data_list=word2vec_util.read_Dataset("sick","C://Users//malla//Downloads//MinHash-master//MinHash-master//")
    train_sick_data=word2vec_util.preprocess(train_sick_data)
    model=word2vec_util.train()
    embedding=word2vec_util.predict_embedding(['Man','is','a','king'])
    print(embedding)
    sentence1='He is a boy'
    sentence2='She is a girl'
    similarity_score=word2vec_util.predict_similarity(sentence1,sentence2)
    print(similarity_score)
    print(word2vec_util.evaluate(model, evaluation_type='sick'))


if __name__ == '__main__':
    main()

'''