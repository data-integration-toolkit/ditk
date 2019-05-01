import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import model_from_json
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class CharacterBasedLSTMModel:
    """ Character-based stacked bi-directional LSTM model
    Based on: `Kuru, Onur, Ozan Arkan Can, and Deniz Yuret. "CharNER: Character-Level Named Entity Recognition.`
    """
    model = ''

    def __init__(self, dataset,**kwargs):
        self.dataset = dataset
        self.model = self.get_model(**kwargs)
        self.metrics = Metrics()

    def load_trained_model(self, weights_path,**kwargs):

        json_file = open(weights_path+'/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_path+"/model.weights.hdf5")

        optimizer = Adam(lr=kwargs.get('learning_rate',0.001),
                         clipnorm=1.0)

        loaded_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', self.non_null_label_accuracy])

        return loaded_model

    def get_model(self,**kwargs):
        num_words = len(self.dataset.alphabet)
        num_labels = len(self.dataset.labels)

        model = Sequential()

        model.add(Embedding(num_words,
                            kwargs.get('embed_size',256),
                            mask_zero=True))
        model.add(Dropout(kwargs.get('input_dropout',0.3)))

        for _ in range(kwargs.get('recurrent_stack_depth',5)):
            model.add(Bidirectional(LSTM(kwargs.get('num_lstm_units',128), return_sequences=True)))

        model.add(Dropout(kwargs.get('output_dropout',0.5)))
        model.add(TimeDistributed(Dense(num_labels, activation='softmax')))

        optimizer = Adam(lr=kwargs.get('learning_rate',0.001),
                         clipnorm=1.0)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', self.non_null_label_accuracy])
        return model

    def fit(self,**kwargs):

        print("Training the model")

        x_train, y_train = self.dataset.get_x_y(dataset_name='train')
        x_dev, y_dev = self.dataset.get_x_y(dataset_name='dev')

        early_stopping = EarlyStopping(patience=kwargs.get('early_stopping',3),
                                       verbose=1)
        checkpointer = ModelCheckpoint(filepath=kwargs.get('checkpoint_dir'+"/model.weights.hdf5","../model/model.weights.hdf5"),
                                       verbose=1,
                                       save_best_only=True)

        self.model.fit(x_train,
                       y_train,
                       batch_size=kwargs.get('batch_size',32),
                       epochs=kwargs.get('max_epochs',10),
                       validation_data=(x_dev, y_dev),
                       shuffle=True,
                       callbacks=[early_stopping, checkpointer])


        self.save_model(**kwargs)


    def save_model(self,**kwargs):

        model_json = self.model.to_json()
        with open(kwargs.get('checkpoint+dir'+"/model.json","../model/model.json"), "w") as json_file:
            json_file.write(model_json)
        print("Saved model in model folder")


    # def fit_generator(self):
    #     train_data_generator = self.dataset.get_x_y_generator(dataset_name='train',
    #                                                           maxlen=self.config.senteCharacterBasedLSTMModelnce_max_length,
    #                                                           batch_size=self.config.batch_size)
    #     dev_data_generator = self.dataset.get_x_y_generator(dataset_name='dev',
    #                                                         maxlen=self.config.sentence_max_length,
    #                                                         batch_size=self.config.batch_size)
    #     early_stopping = EarlyStopping(patience=self.config.early_stopping,
    #                                    verbose=1)
    #
    #     self.model.fit_generator(train_data_generator,
    #                              steps_per_epoch=self.dataset.num_train_docs / self.config.batch_size,
    #                              epochs=self.config.max_epochs,
    #                              validation_data=dev_data_generator,
    #                              validation_steps=self.dataset.num_dev_docs / self.config.batch_size,
    #                              callbacks=[early_stopping]
    #                              )

    def evaluate(self,**kwargs):
        x_test, y_test = self.dataset.get_x_y(dataset_name='test')
        scores = self.model.evaluate(x_test, y_test, batch_size=kwargs.get('batch_size',32))
        print("\n")

        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        print("\n%s: %.2f%%" % (self.model.metrics_names[2], scores[2] * 100))
        print("\n%s: %.2f%%" % (self.model.metrics_names[0], scores[0] * 100))


    # def evaluate_generator(self):
    #     test_data_generator = self.dataset.get_x_y_generator(dataset_name='test',
    #                                                          maxlen=self.config.sentence_max_length,
    #                                                          batch_size=self.config.batch_size)
    #
    #     self.model.evaluate_generator(test_data_generator, steps=self.dataset.num_test_docs / self.config.batch_size)

    def predict_str(self, s,saved_model):
        """ Get model prediction for a string
        :param s: string to get named entities for
        :return: a list of len(s) tuples: [(character, predicted-label for character), ...]
        """
        x = self.dataset.str_to_x(s,200)
        predicted_classes = self.predict_x(x,saved_model)
        chars = self.dataset.x_to_str(x)[0]
        labels = self.dataset.y_to_labels(predicted_classes)[0]

        """
        Converting char to words and 
        """

        word = ""
        labels_per_word = []
        predictions = []


        for i in range(len(chars)):
            if(len(chars[i].strip()) and chars[i].strip()!='<PAD>'):
                word+=chars[i]
                labels_per_word.append(labels[i])
            else:#word over

                frequent_label = self.most_frequent(labels_per_word)
                predictions.append((None,None,word,frequent_label))
                word = ''
                labels_per_word = []
                if(chars[i].strip()=='<PAD>'):
                    break


        return predictions


    def most_frequent(self,List):
        return max(set(List), key=List.count)


    def predict_x(self, x,saved_model):
        return saved_model.predict(x, batch_size=1)

    @staticmethod
    def non_null_label_accuracy(y_true, y_pred):
        """Calculate accuracy excluding null-label targets (index 0).
        Useful when the null label is over-represented in the data, like in Named Entity Recognition tasks.

        typical y shape: (batch_size, sentence_length, num_labels)
        """

        y_true_argmax = K.argmax(y_true, -1)  # ==> (batch_size, sentence_length, 1)
        y_pred_argmax = K.argmax(y_pred, -1)  # ==> (batch_size, sentence_length, 1)

        y_true_argmax_flat = tf.reshape(y_true_argmax, [-1])
        y_pred_argmax_flat = tf.reshape(y_pred_argmax, [-1])

        non_null_targets_bool = K.not_equal(y_true_argmax_flat, K.zeros_like(y_true_argmax_flat))
        non_null_target_idx = K.flatten(K.cast(tf.where(non_null_targets_bool), 'int32'))

        y_true_without_null = K.gather(y_true_argmax_flat, non_null_target_idx)
        y_pred_without_null = K.gather(y_pred_argmax_flat, non_null_target_idx)

        mean = K.mean(K.cast(K.equal(y_pred_without_null,
                                     y_true_without_null),
                             K.floatx()))

        # If the model contains a masked layer, Keras forces metric output to have same shape as y:
        fake_shape_mean = K.ones_like(y_true_argmax, K.floatx()) * mean
        return fake_shape_mean

    def get_custom_objects(self):
        return {'non_null_label_accuracy': self.non_null_label_accuracy}


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return





