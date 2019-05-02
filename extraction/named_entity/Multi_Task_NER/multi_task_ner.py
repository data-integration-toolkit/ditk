#from ditk import NER
#
# The parent class is located at: https://github.com/AbelJJohn/csci548sp19projectner
#
import numpy as np
seed_number = 1337
np.random.seed(seed_number)

from extraction.named_entity.Multi_Task_NER.common import utilities as utils
from extraction.named_entity.Multi_Task_NER.common import representation as rep
from extraction.named_entity.Multi_Task_NER.models import network
from extraction.named_entity.Multi_Task_NER.models import crf
from extraction.named_entity.Multi_Task_NER.settings import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import model_from_json


class multi_task_ner():
    def __init__(self):
        pass
    def read_dataset(self,file_dict,dataset_name = '', *args, **kwargs):
        #read sentences, pos tags, and entity tags from files, generate datasets for training, validation and test
        #input:
        #   fileNames: list of file  paths

        #return:
        #   data: train data, validation data, test data

        #implementation:
        #   read data and convert data into proper format
        #   write data in proper format into new files
        #   read data from new files
        '''
        _DATA_DIR = '/Users/boyuzhang/NER-WNUT17-master/dataset/'
        if (dataset_name == 'conll'):
            TRAIN = _DATA_DIR + 'train.conll'
            DEV = _DATA_DIR + 'dev.conll'
            TEST = _DATA_DIR + 'test.conll'

        else:
            TRAIN = _DATA_DIR + 'train.ontonotes'
            DEV = _DATA_DIR + 'dev.ontonotes'
            TEST = _DATA_DIR + 'test.ontonotes'
        TRAIN_PREPROC_URL = TRAIN + '.preproc.url'
        TRAIN_PREPROC_URL_POSTAG = TRAIN + '.preproc.url.postag'

        DEV_PREPROC_URL = DEV + '.preproc.url'
        DEV_PREPROC_URL_POSTAG = DEV+ '.preproc.url.postag'

        TEST_PREPROC_URL = TEST + '.preproc.url'
        TEST_PREPROC_URL_POSTAG = TEST + '.preproc.url.postag'
        '''

        utils.preprocess_data(file_dict)

        train_file = file_dict['train']

        predix = ''
        if train_file.rfind('/') >= 0:
            predix = train_file[:train_file.rfind('/')] + '/'

        TRAIN_PREPROC_URL = predix+'train.conll.preproc.url'
        DEV_PREPROC_URL = predix+'dev.conll.preproc.url'
        TEST_PREPROC_URL = predix+'test.conll.preproc.url'

        TRAIN_PREPROC_URL_POSTAG  = predix+'train.conll.preproc.url.postag'
        DEV_PREPROC_URL_POSTAG = predix+'dev.conll.preproc.url.postag'
        TEST_PREPROC_URL_POSTAG = predix+'test.conll.preproc.url.postag'

        tweets_train, labels_train = utils.read_file_as_lists(TRAIN_PREPROC_URL)
        tweets_dev, labels_dev = utils.read_file_as_lists(DEV_PREPROC_URL)
        tweets_test, labels_test = utils.read_file_as_lists(TEST_PREPROC_URL)

        # Combining train and dev to account for different domains
        tweets_train += tweets_dev
        labels_train += labels_dev

        index2category = []
        for labels in labels_train:
            for label in labels:
                index2category.append(label)
        for labels in labels_test:
            for label in labels:
                index2category.append(label)

        index2category = list(set(index2category))
        self.index2category = list(set(index2category))
        print("nunber of category: " + str(len(index2category)))
        print(index2category)

        postag_train, postag_test = utils.read_and_sync_postags(tweets_train, tweets_test,TRAIN_PREPROC_URL_POSTAG,DEV_PREPROC_URL_POSTAG,TEST_PREPROC_URL_POSTAG)

        print("Loading twitter embeddings...")
        self.twitter_embeddings, self.word2index = utils.read_twitter_embeddings(tweets_train + tweets_test)

        ## GAZETTERS
        print("Loading gazetteers embeddings...")
        self.gaze_embeddings, self.gaze2index = utils.read_gazetteer_embeddings()

        if len(self.index2category) != 6:
            self.gaze = False
        else:
            self.gaze = True
        self.gaze = True
        return [tweets_train, labels_train, tweets_test, labels_test, postag_train, postag_test]

    def generating_encoding(self, data, *args, **kwargs):
        # generating encoding with data to train the model
        # input:
        #   data: data extracted from datasets
        # return:
        #   encoding: encoding of data

        # implementation:
        #   encode words
        #   encode labels
        #   generate Pos Tags if not have
        #   encode Pos Tags
        #   encode orthograph
        #   en code GAZETTEERS
        tweets_train = data[0]
        labels_train = data[1]
        tweets_test = data[2]
        labels_test = data[3]
        postag_train = data[4]
        postag_test = data[5]
        print("Generating encodings...")
        ## WORDS (X)
        self.radius = 1
        x_word_twitter_train = rep.encode_tweets(self.word2index, tweets_train, self.radius)
        x_word_twitter_test = rep.encode_tweets(self.word2index, tweets_test, self.radius)

        ## LABELS (Y)
        y_bin_train = rep.encode_bin_labels(labels_train)
        y_cat_train = rep.encode_cat_labels(labels_train,self.index2category)

        ## POS TAGS
        self.index2postag = [PAD_TOKEN] + utils.get_uniq_elems(postag_train + postag_test)
        x_postag_train = rep.encode_postags(self.index2postag, postag_train, self.radius)
        x_postag_test = rep.encode_postags(self.index2postag, postag_test, self.radius)

        ## ORTHOGRAPHY
        self.ortho_dim = 30
        self.ortho_max_length = 20
        x_ortho_train = rep.encode_orthography(tweets_train, self.ortho_max_length)
        x_ortho_test = rep.encode_orthography(tweets_test, self.ortho_max_length)

        ## GAZETTEERS
        if self.gaze:
            x_gaze_train = rep.encode_gazetteers(self.gaze2index, tweets_train, self.radius)
            x_gaze_test = rep.encode_gazetteers(self.gaze2index, tweets_test, self.radius)
            return [x_word_twitter_train,x_word_twitter_test,y_bin_train,y_cat_train,x_postag_train,x_postag_test,x_ortho_train,x_ortho_test,x_gaze_train,x_gaze_test]
        else:
            return [x_word_twitter_train, x_word_twitter_test, y_bin_train, y_cat_train, x_postag_train, x_postag_test,
                    x_ortho_train, x_ortho_test]

    def save_model(self,file = ''):

        model_json = self.mtl_network.to_json()
        with open(file +"model.json", "w") as json_file:
            json_file.write(model_json)
        self.mtl_network.save_weights(file + "model.h5")


    def load_model(self,file = ''):
        json_file = open(file+'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.mtl_network = model_from_json(loaded_model_json)
        # load weights into new model
        self.mtl_network.load_weights(file+"model.h5")


    def train(self, data, *args, **kwargs):
        # build model, train the model with loaded embeddings and encoded data
        # input:
        #   data: encoded data
        # return:
        #   None, model is stored internally

        # implementation:
        #   load pre-trained embeddings
        #   build nureal network
        #   train the neural network
        self.char_inputs, self.char_encoded = network.get_char_cnn(self.ortho_max_length, len(rep.index2ortho), self.ortho_dim,
                                                         'char_ortho')
        self.word_inputs, self.word_encoded = network.get_word_blstm(len(self.index2postag), self.twitter_embeddings, window=self.radius * 2 + 1,
                                                           word_dim=100)
        print(len(self.index2category))
        if self.gaze:
            self.gaze_inputs, self.gaze_encoded = network.get_gazetteers_dense(self.radius * 2 + 1, self.gaze_embeddings)

            self.mtl_network = network.build_multitask_bin_cat_network(len(self.index2category),  # number of category classes
                                                              self.char_inputs, self.char_encoded,  # char component (CNN)
                                                              self.word_inputs, self.word_encoded,  # word component (BLSTM)
                                                              self.gaze_inputs, self.gaze_encoded,self.gaze)  # gazetteer component (Dense)
        else:
            self.mtl_network = network.build_multitask_bin_cat_network(len(self.index2category),
                                                                       # number of category classes
                                                                       self.char_inputs, self.char_encoded,
                                                                       # char component (CNN)
                                                                       self.word_inputs, self.word_encoded,
                                                                       # word component (BLSTM)
                                                                       [],
                                                                    [],self.gaze)  # gazetteer component (Dense)
        self.mtl_network.summary()

        x_word_twitter_train = data[1][0]
        x_postag_train = data[1][4]
        x_ortho_train = data[1][6]
        if self.gaze:
            x_gaze_train = data[1][8]

        train_word_values = [x_word_twitter_train, x_postag_train]
        train_char_values = [x_ortho_train]
        if self.gaze:
            train_gaze_values = [x_gaze_train]
        y_bin_train = data[1][2]
        y_cat_train = data[1][3]

        if self.gaze:
            self.x_train_samples = train_gaze_values + train_char_values + train_word_values
        else:
            self.x_train_samples = train_char_values + train_word_values

        self.y_train_samples = {'bin_output': y_bin_train, 'cat_output': y_cat_train}

        hist = network.train_multitask_net_with_split(self.mtl_network, self.x_train_samples, self.y_train_samples)
        return hist

    def predict(self, data, *args, **kwargs):
        # predict the label of test data
        # input:
        #   data: data used to conduct prediction
        # return:
        #   prediction results

        #implementation:
        #   conduct prediction
        if self.gaze:
            x_gaze_test = data[1][9]
        x_ortho_test = data[1][7]
        x_word_twitter_test = data[1][1]
        x_postag_test = data[1][5]
        labels_train = data[0][1]

        if self.gaze:
            x_test = [x_gaze_test, x_ortho_test, x_word_twitter_test, x_postag_test]
            inputs = self.gaze_inputs + self.char_inputs + self.word_inputs
        else:
            x_test = [x_ortho_test, x_word_twitter_test, x_postag_test]
            inputs = self.char_inputs + self.word_inputs

        decoded_predictions_NN = network.predict(self.mtl_network, inputs, x_test, self.index2category)

        # Saving predictions in format: token\tlabel\tprediction
        #utils.save_predictions(NN_PREDICTIONS, tweets_test, labels_test, decoded_predictions)

        fextractor = network.create_model_from_layer(self.mtl_network, layer_name='common_dense_layer')
        crf.train_with_fextractor(fextractor, self.x_train_samples, labels_train)

        decoded_predictions_crf = crf.predict_with_fextractor(fextractor, x_test)
        return [decoded_predictions_NN,decoded_predictions_crf]

    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        # evaluate the predictons with the graound truths
        # inputs:
        #   predictions: the results returned by pridict() function
        #   groundTruths: the targets of the prediction results
        # return:
        #   evalution results

        #implementation:
        #   evaluate predictions with graoundTruths
        labels_test = groundTruths

        decoded_predictions_NN = predictions[0]

        evaluation_NN = precision_recall_fscore_support(utils.flatten(labels_test), decoded_predictions_NN, average=None)

        print("NN Classification Report\n")
        print(classification_report(utils.flatten(labels_test), decoded_predictions_NN))
        print()
        print()
        print("NN Confusion Matrix\n")
        print(confusion_matrix(utils.flatten(labels_test), decoded_predictions_NN))

        decoded_predictions_CRF = predictions[1]
        evaluation_CRF = precision_recall_fscore_support(utils.flatten(labels_test), decoded_predictions_CRF, average=None)
        print("CRF Classification Report\n")
        print(classification_report(utils.flatten(labels_test), decoded_predictions_CRF))
        print()
        print()
        print("CRF Confusion Matrix\n")
        print(confusion_matrix(utils.flatten(labels_test), decoded_predictions_CRF))
        return [evaluation_NN,evaluation_CRF]

    def convert_ground_truth(self, data, *args, **kwargs):
        # convert data into the proper format
        # inputs:
        #   data: data in proper format
        # return:
        #   labels in the proper format

        #implementation:
        #   extract labels from data
        #   convert labels in the proper format
        return data[3]

    def main(self,fileNames):

        data = self.read_dataset(fileNames,"dataset")
        embedding = self.generating_encoding(data)
        self.train([data, embedding])
        decoded_predictions = self.predict([data, embedding])
        ground_truth = self.convert_ground_truth(data)[0]
        NN_pre = decoded_predictions[0]
        #ground_truth = ground_truth
        test_words = data[2][0]

        file_path = 'output.txt'
        file = open(file_path, 'w')
        for i in range(len(test_words)):
            file.write(str(test_words[i])+" "+str(ground_truth[i])+" "+ str(NN_pre[i])+'\n')

        file.close()

        target = self.convert_ground_truth(data)
        evaluation_result = self.evaluate(decoded_predictions, target)

        for i in range(len(evaluation_result)):
            if i == 0:
                print('NN precision: ', str(evaluation_result[i][0]))
                print('NN recall: ', str(evaluation_result[i][1]))
                print('NN f1: ', str(evaluation_result[i][2]))
            else:
                print('CRF precision: ', str(evaluation_result[i][0]))
                print('CRF recall: ', str(evaluation_result[i][1]))
                print('CRF f1: ', str(evaluation_result[i][2]))
        return file_path

