from keras import backend as K
from src.dataset import CharBasedNERDataset
from src.model import CharacterBasedLSTMModel
from src.ner import Ner
from src.datasetProperties import datasetProperties

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)




class CharNER(Ner):

    def __init__(self):
        self.dataset = ''

    def convert_ground_truth(self, data, *args, **kwargs):

        groundTruths = []
        test_data = data['test']

        for word in test_data:
            if(len(word)):
                row_tuple = (None,None,word[0],word[3].lower())
                groundTruths.append(row_tuple)
            else:
                groundTruths.append('')

        return groundTruths

    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):

        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()

                raw_data = raw_data[2:]


                for i, line in enumerate(raw_data):
                    if len(line.strip()) > 0:
                        raw_data[i] = line.strip().split()
                    else:
                        raw_data[i] = list(line)
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)

        print("Data read successfully")

        return data

        """
        file_dict = {"train":"test.train.ner", "test":"test.train.ner","dev":"test.train.ner"}
        data = read_dataset(file_dict)
        i = 0
        """

    def train(self, data, *args, **kwargs):


        if 'checkpoint_dir' not in kwargs:
            return ValueError("Insert checkpoint_dir to make the model run and save")


        datasetProperties(data)

        self.dataset = CharBasedNERDataset(data)

        """
        Setting up the model with all its parameters 
        """
        model = CharacterBasedLSTMModel(self.dataset,**kwargs)

        """
        Printing the paramters the model gets trained on
        """

        print("Input Dropout Rate:- "+str(kwargs.get('input_dropout',0.3)))
        print("Output Dropout Rate:- "+str(kwargs.get('output_dropout',0.5)))
        print("Recurrent Stack Depth:- "+str(kwargs.get('recurrent_stack_depth',5)))
        print("Batch Size:- "+str(kwargs.get('batch_size',32)))
        print("Max Epochs:- "+str(kwargs.get('max_epochs',10)))
        print("Learning Rate:- "+str(kwargs.get('learning_rate',0.001)))
        print("Embed Size:- " +str(kwargs.get('embed_size',256)))
        print("Number of LSTM units:- "+str(kwargs.get('num_lstm_units',128)))
        print("Early Stopping:- "+str(kwargs.get('early_stopping',2)))


        """
        Training the model
        """
        model.fit(**kwargs)


    def predict(self, data, *args, **kwargs):



        model_object = CharacterBasedLSTMModel(self.dataset,**kwargs)

        if 'model_dir' not in kwargs:
            raise ValueError("Please give value of model_dir in kwargs")

        print("Loading model for predicting")

        saved_model = self.load_model(kwargs.get('model_dir'),**kwargs)

        sentences = list()
        true_labels =[]
        words = []
        #convert data into sentences
        a = []
        for word in data:
            if(len(word)):
                a.append(word)
            else:
                if(len(a)):
                    ws = ""
                    for e1 in a:
                        ws+=e1[0]+" "
                        true_labels.append(e1[3].lower())
                        words.append(e1[0])
                    sentences.append(ws[0:len(ws)-1])
                    words.append('')
                    true_labels.append('')
                a = []

        if(len(a)):
            ws = ""
            for e1 in a:
                ws += e1[0] + " "
                true_labels.append(e1[3].lower())
                words.append(e1[0])
            sentences.append(ws[0:len(ws) - 1])


        predictions = list()

        for d in sentences:
            """
            Make predictions using the saved model
            """
            string_predictions = model_object.predict_str(d,saved_model)

            for word in string_predictions:
                predictions.append(word)

            predictions.append('')


        predictions = predictions[:-1]

        #save to file

        with open('../output.txt','w+') as f:

            f.writelines('WORD TRUE_LABEL PRED_LABEL\n')
            f.writelines('\n')

            for x in range(0,len(predictions)):
                if(len(predictions[x])):
                    f.writelines(' '.join([str(words[x]),str(true_labels[x]),str(predictions[x][3].strip())])+'\n')
                else:
                    f.writelines('\n')



        output_file_path = "../output.txt"

        return predictions,output_file_path


    def evaluate(self, predictions, groundTruths, *args, **kwargs):

        """
        calculation of f1,precision,recall
        :param predictions:
        :param groundTruths:
        :param args:
        :param kwargs:
        :return: (precision , recall , f1))
        """

        preds = []
        grounds = []

        for i in range(0,len(predictions)):
            if(len(predictions[i])):
                preds.append(predictions[i][3])
                grounds.append(groundTruths[i][3])

        preds_array = np.asarray(preds)
        grounds_array = np.asarray(grounds)

        accuracy = accuracy_score(grounds_array, preds_array)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(grounds_array, preds_array,average='weighted',labels=np.unique(grounds_array))
        # print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(grounds_array, preds_array,average='weighted',labels=np.unique(grounds_array))
        # print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(grounds_array, preds_array,average='weighted',labels=np.unique(grounds_array))
        # print('F1 score: %f' % f1)

        return precision,recall,f1



    def load_model(self,weights_path,**kwargs):

        print("Loading the model from "+weights_path)
        model = CharacterBasedLSTMModel(self.dataset)
        x = model.load_trained_model(weights_path,**kwargs)
        print("Model loaded successfully")

        return x



    def save_model(self,**kwargs):

        model = CharacterBasedLSTMModel(self.dataset)
        model.save_model(**kwargs)

        if 'checkpoint_dir' not in kwargs:
            raise ValueError('Provide checkpoint_dir where to save the model')

        print("Model saved at given location :- "+kwargs.get('checkpoint_dir'))

        pass


    def f1(self,y_true, y_pred):

        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    def main(self,inputFile):

        file_Dict = dict()
        file_Dict['train'] = inputFile
        file_Dict['test'] =inputFile
        file_Dict['dev'] = inputFile

        # reading data
        data_dict = self.read_dataset(file_Dict, 'conll3')

        print("Training Data ")
        print(data_dict['train'])

        print("Testing Data ")
        print(data_dict['test'])

        print("Dev Data ")
        print(data_dict['dev'])

        # training model
        self.train(data_dict,
                         checkpoint_dir="../model")

        # evaluate model after train
        """
        get ground truths for evaluate
        get predictions for test
        calculate f1 using evaluate
        """
        ground_truths = self.convert_ground_truth(data_dict)
        predicted_values,output_file_path = self.predict(data_dict['test'],
                                              model_dir="../model")
        precision,recall,f1 = self.evaluate(predicted_values, ground_truths)

        print("Precision : ",precision)
        print("Recall : ",recall)
        print("F1 : ",f1)

        # Saving the model
        self.save_model(
            checkpoint_dir="../model")

        # Loading the model
        model = self.load_model("../model")



        return output_file_path