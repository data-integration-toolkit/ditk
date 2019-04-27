# -*- coding: utf-8 -*-
import datetime
import os
import sys
from time import time

from keras.optimizers import Adadelta

from modules.log_config import LOG
from modules.configs import *
from modules.input_data import ProcessInputData
from modules.datasets import *
from modules.model import init_model
from modules.embedding import load_embedding_matrix
from modules.result_data import create_output
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_similarity import TextSemanticSimilarity
import pickle
from keras.models import Model, load_model

class Semantic_Neural_Network(TextSemanticSimilarity):

    def __init__(self):
        self.embedding_matrix = []
        self.database_name=''
        self.total_duration= 0
        self.timestamp = 0
        self.test_dataframe=[]
    @classmethod
    def read_dataset(self, fileNames, databaseName, *args, **kwargs):
        """
        Reads a dataset that is a CSV/Text File.

        Args:
            fileName : With it's absolute path
            databaseName : Supports SICK, SemEval2014 and SemEval2017 datafile format along with a Generic format(s1,s2,relatedness_score)
        Returns:
            training_data_list : Data in the format to be used by train() module.
        Raises:
            None
        """
        # parse files to obtain the output
        if(len(databaseName)>0):
            self.database_name=databaseName
        else:
            self.database_name='Generic'
        if databaseName in 'SICK':
            train_df = SICKDataset(DATA_PATH+fileNames[0]).data_frame()
            val_df = SICKDataset(DATA_PATH+fileNames[1]).data_frame()
            test_df = SICKDataset(DATA_PATH+fileNames[2]).data_frame()
            train_df = pd.concat([train_df, val_df])
        elif databaseName in 'SemEval2014':
            train_df = SemEval2014Dataset(DATA_PATH+fileNames[0]).data_frame()
            val_df = SemEval2014Dataset(DATA_PATH+fileNames[1]).data_frame()
            test_df = SemEval2014Dataset(DATA_PATH+fileNames[2]).data_frame()
            train_df = pd.concat([train_df, val_df])
        elif databaseName in 'SemEval2017':
            train_df = SemEval2017Dataset(DATA_PATH+fileNames[0]).data_frame()
            val_df = SemEval2017Dataset(DATA_PATH+fileNames[1]).data_frame()
            test_df = SemEval2017Dataset(DATA_PATH+fileNames[2]).data_frame()
            train_df = pd.concat([train_df, val_df])
        else:
            train_df = GenericDataset(DATA_PATH+fileNames[0]).data_frame()
            val_df = GenericDataset(DATA_PATH+fileNames[1]).data_frame()
            test_df = GenericDataset(DATA_PATH+fileNames[2]).data_frame()
            train_df = pd.concat([train_df, val_df])

        self.test_dataframe = test_df
        # ===============================
        # PREPARE INPUT DATA - PREPROCESS
        # ===============================
        process = ProcessInputData()
        read_dataset_output = dict()

        train_data = process.prepare_data([train_df], databaseName)
        read_dataset_output['train_input'] = train_data[0]
        test_data = process.prepare_data([test_df], databaseName)
        read_dataset_output['test_input'] = test_data[0]

        read_dataset_output['max_sentence_length'] = process.max_sentence_length
        read_dataset_output['vocab_size'] = process.vocabulary_size + 1
        read_dataset_output['word_index'] = process.word_index
        return read_dataset_output

    @classmethod
    def get_embedding_matrix(self,word_index):
        print(EMBEDDING_BINARY)
        self.embedding_matrix = load_embedding_matrix(self.database_name, word_index)

    @classmethod
    def train(self, read_dataset_output,*args, **kwargs):  # <--- implemented PER class

        # =========================================
        #     CREATING MODEL
        # =========================================
        self.get_embedding_matrix(read_dataset_output['word_index'])
        model = init_model(read_dataset_output['max_sentence_length'], self.embedding_matrix, DROPOUT, RECURRENT_DROPOUT, read_dataset_output['vocab_size'])
        gradient_clipping_norm = 1.6
        optimizer = Adadelta(lr=LR, clipnorm=gradient_clipping_norm)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # =========================================
        # ============== TRAIN MODEL ==============
        # =========================================
        start_train = time()

        # ============= TRAIN =============
        LOG.info("START TRAIN")
        training_time = time()
        train_input = read_dataset_output['train_input']
        test_input = read_dataset_output['test_input']

        train_history = model.fit([train_input.x1, train_input.x2], train_input.y,
                                  epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE,
                                  validation_split=0.1, verbose=FIT_VERBOSE)

        duration = datetime.timedelta(seconds=time() - training_time)
        LOG.info("\nTraining time finished.\n{} epochs in {}".format(TRAIN_EPOCHS, duration))
        self.total_duration = datetime.timedelta(seconds=time() - start_train)

        # ======= STORE TRAINED MODEL ======
        LOG.info('Saving Model')
        self.timestamp = (int(round(time() * 1000)))
        model_file = 'model_%s.h5' % self.database_name
        model_file_path = os.path.join(RESULTS_DIR, model_file)
        model.save(model_file_path)

        predication_output = model.predict([test_input.x1, test_input.x2])
        return predication_output

    @classmethod
    def predict(self, data_X, data_Y, database_name ='Generic', *args, **kwargs):
        """
        Predicts the similarity score on the given input data(2 sentences). Assumes model has been trained with train()
        Reads the tokenizer and model created in the train module().
        Args:
            data_X: Sentence 1(Non Tokenized).
            data_Y: Sentence 2(Non Tokenized)

        Returns:
            prediction_score: Similarity Score ( Float )

        Raises:
            None
        """
        TOKENIZER = BASE_PATH +"tokenizers/tokenizer_%s.pickle" %(database_name)
        MODEL_FILE = RESULTS_DIR +"model_%s.h5" %(database_name)
        with open(TOKENIZER, 'rb') as handle:
            tokenizer = pickle.load(handle)

        preprocess = ProcessInputData(tokenizer=tokenizer)
        x1, x2 = preprocess.get_input_from_collection([data_X], [data_Y], 100)
        model = load_model(MODEL_FILE)
        a = model.predict([x1, x2])
        return a

    @classmethod
    def evaluate(self, actual_values, predicted_values, *args, **kwargs):
        """
        Returns the correlation score(0-1) between the actual and predicted similarity scores

        Args:
            actual_values : List of actual similarity scores
            predicted_values : List of predicted similarity scores

        Returns:
            correlation_coefficient : Value between 0-1 to show the correlation between the values(actual and predicted)

        Raises:
            None
        """
        # =========================
        # EVALUATE
        # =========================
        model_file = 'model_%s.h5' % self.database_name
        LOG.info('Creating result file')
        self.pearson_metric_score = create_output(self.database_name, self.test_dataframe, predicted_values, actual_values, self.total_duration, model_file,
                                                  self.timestamp, obs=str(sys.argv))
        return self.pearson_metric_score



