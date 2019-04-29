"""
Build a NERModel
"""

from ChainCRF import ChainCRF, create_custom_objects
import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)
from keras.layers import Dense, Dropout, Add, Input, Reshape, Lambda, Concatenate
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional, TimeDistributed
from keras.layers import Conv1D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, RMSprop
import keras
print keras.__version__

import numpy as np
import cPickle as pickle


class NERModel(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        self.parameters_path = model_path + '_parameters.pkl'
        if not parameters['reload'] or ('override_parameters' in parameters and parameters['override_parameters']):  # init model using parameters
            self.parameters = parameters
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                pickle.dump(parameters, f)
        else:  # reload a model
            # Load the parameters and the mappings from disk
            # Override parameters
            with open(self.parameters_path, 'rb') as f:
                self.parameters = pickle.load(f)

        # don't change reload
        self.parameters['reload'] = parameters['reload']
        if 'tag' in parameters:
            self.parameters['tag'] = parameters['tag']

        self.model = self.build_model()
        self.parameters['ner_model'] = self


    def build_model(self):
        """
        Build CNN character model with optional dropout
        """
        params = self.parameters
        # reload previous model
        if params['reload']:
            if params['crf_on_top']:
                model = load_model(params['model_path'], custom_objects=create_custom_objects())
            else:
                model = load_model(params['model_path'])
            print 'Reloading model...'
        else:
            # Input params
            max_chars_in_sample = params['max_chars_in_sample']
            use_bytes = params['use_bytes']

            # BPE params
            use_bpe = params['use_bpe']
            use_bpe_embeddings = params['use_bpe_embeddings']
            if use_bpe_embeddings:
                bpe_vocab_dict = params['bpe_vocab_dict']
            bpe_input_dim = params['bpe_embeddings_dim']

            # tok params
            use_tokenization = params['use_tokenization']

            # Char embedding params
            input_dim = params['embedding_input_dim']
            output_dim = params['embedding_output_dim']

            # Word embedding params
            word_input_dim = params['word_embeddings_dim']
            use_word_embeddings = params['use_word_embeddings']
            if use_word_embeddings:
                word_vocab_dict = params['word_vocab_dict']

            # Dropout params
            dropout = params['dropout']

            # CNN params
            filters = params['cnn_filters']
            kernel_size = params['cnn_kernel_size']
            padding = params['cnn_padding']
            act = params['cnn_act']
            num_byte_layers = params['num_byte_layers']
            residual = params['residual']
            skip_residuals = params['skip_residuals']
            byte_layer_for_embed = params['byte_layer_for_embed']
            layer_for_bytes = params['layer_for_bytes']  # blstm or cnn

            # Dense params
            units = params['num_iobes_tags']
            final_act = params['dense_final_act']

            # Compile params
            lr = params['lr']
            if params['optimizer'] == 'adam':
                optimizer = Adam(lr=lr, clipnorm=1.0)
            elif params['optimizer'] == 'rmsprop':
                optimizer = RMSprop(lr=lr, clipnorm=1.0)
            else:
                optimizer = SGD(lr=lr, clipnorm=1.0)

            # BLSTM on top
            blstm_on_top = params['blstm_on_top']
            lstm_act = params['lstm_act']
            lstm_units = params['lstm_units']

            # CRF on top
            crf_on_top = params['crf_on_top']

            # functional implementation
            y = None
            y_prev = None  # for residual connections
            skip_residuals_counter = 0

            # char embedding
            num_feats = sum([1 for x in [use_bytes, use_bpe, use_tokenization] if x])
            num_bfeats = sum([1 for x in [use_bytes, use_bpe] if x])
            if num_feats > 0:
                inputs = Input(shape=(max_chars_in_sample, num_feats), name='byte_input')
                to_concat = []

                if use_bpe or use_bytes:
                    if use_tokenization:

                        def get_all_but_last_feature(layer):
                            return layer[:, :, :-1]

                        x = Lambda(get_all_but_last_feature, output_shape=(max_chars_in_sample, num_bfeats), name='main_feats')(inputs)
                    else:
                        x = inputs
                    x = Embedding(input_dim=input_dim, output_dim=output_dim)(x)
                    x = Reshape((max_chars_in_sample, output_dim * num_bfeats))(x)  # reshape from (sequence_len, embedding_dim, num_feats) into (sequence_len, embedding_dim*num_feats)
                    to_concat.append(x)

                if use_tokenization:

                    def get_last_feature(layer):
                        return layer[:, :, -1]

                    num_iobes_toks = 5
                    x = Lambda(get_last_feature, output_shape=(max_chars_in_sample, 1), name='tok_feats')(inputs)
                    x = Embedding(input_dim=num_iobes_toks, output_dim=num_iobes_toks)(x)  # IOBES for tokenization
                    x = Reshape((max_chars_in_sample, num_iobes_toks))(x)
                    to_concat.append(x)

                if len(to_concat) > 1:
                    x = Concatenate()(to_concat, axis=2)
                x = Dropout(rate=dropout)(x)  # dropout after embedding layer
            else:
                x = None

            # word embedding
            if use_word_embeddings:
                n_symbols = len(word_vocab_dict)  # space, unknown, and padding included
                embedding_weights = np.zeros((n_symbols, word_input_dim))
                for word_num, embed in word_vocab_dict.values():
                    embedding_weights[word_num, :] = embed

                inputs2 = Input(shape=(max_chars_in_sample, 1), name='word_input')
                x2 = Embedding(input_dim=n_symbols,
                               output_dim=word_input_dim,
                               trainable=False)
                x2.build((None,))
                x2.set_weights([embedding_weights])
                x2 = x2(inputs2)
                x2 = Reshape((max_chars_in_sample, word_input_dim))(x2)
                x2 = Dropout(rate=dropout)(x2)
            elif use_bpe_embeddings:
                n_symbols = len(bpe_vocab_dict)
                embedding_weights = np.zeros((n_symbols, bpe_input_dim))
                for bpe_num, embed in bpe_vocab_dict.values():
                    embedding_weights[bpe_num, :] = embed

                inputs2 = Input(shape=(max_chars_in_sample, 1), name='bpe_input')
                x2 = Embedding(input_dim=n_symbols,
                               output_dim=bpe_input_dim,
                               trainable=params['trainable_bpe_embeddings'])
                x2.build((None,))
                x2.set_weights([embedding_weights])
                x2 = x2(inputs2)
                x2 = Reshape((max_chars_in_sample, bpe_input_dim))(x2)
                x2 = Dropout(rate=dropout)(x2)
            else:
                x2 = None

            if layer_for_bytes == 'cnn':
                # if running embedding features through CNN, process inputs here
                if byte_layer_for_embed:
                    if x is not None and x2 is not None:
                        x = Concatenate()([x, x2])
                    elif x2 is not None:
                        x = x2
                    elif x is not None:
                        x = x
                    else:
                        print 'No input features!'
                        exit()

                while num_byte_layers > 0 and x is not None:
                    if residual and y_prev is not None:
                        if skip_residuals:
                            if skip_residuals_counter:
                                x = Add()([x, y_prev])
                                skip_residuals_counter = 0
                            else:
                                skip_residuals_counter = 1
                        else:
                            x = Add()([x, y_prev])

                    y_prev = y
                    y = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=act)(x)
                    if dropout:
                        x = Dropout(rate=dropout)(y)
                    else:
                        x = y
                    num_byte_layers -= 1
            elif layer_for_bytes == 'blstm':
                # if running embedding features through byte layer, process inputs here

                if byte_layer_for_embed:
                    if x is not None and x2 is not None:
                        x = Concatenate()([x, x2])
                    elif x2 is not None:
                        x = x2
                    elif x is not None:
                        x = x
                    else:
                        print 'No input features!'
                        exit()

                while num_byte_layers > 0 and x is not None:
                    x = Bidirectional(LSTM(units=filters, activation=lstm_act, return_sequences=True))(x)
                    if dropout:
                        x = Dropout(rate=dropout)(x)
                    num_byte_layers -= 1

            # if not running embedding features through CNN, process inputs here
            if not byte_layer_for_embed:
                if x is not None and x2 is not None:
                    x = Concatenate()([x, x2])
                elif x2 is not None:
                    x = x2
                elif x is not None:
                    x = x
                else:
                    print 'No input features!'
                    exit()

            # BLSTM layer on top
            if blstm_on_top:
                if layer_for_bytes == 'blstm':
                    x = Dense(units=filters*4, activation=act)(x)
                else:
                    x = Dense(units=filters, activation=act)(x)
                x = Bidirectional(LSTM(units=lstm_units, activation=lstm_act, return_sequences=True))(x)

            # CRF layer on top
            if crf_on_top:
                if blstm_on_top:
                    x = TimeDistributed(Dense(units=units, activation=act))(x)
                else:
                    x = Dense(units=units, activation=act)(x)
                crf = ChainCRF()
                predictions = crf(x)
                loss = crf.loss
            else:
                if blstm_on_top:
                    predictions = TimeDistributed(Dense(units=units, activation=final_act))(x)
                else:
                    predictions = Dense(units=units, activation=final_act)(x)
                loss = 'categorical_crossentropy'

            actual_inputs = None
            if use_bytes or use_bpe or use_tokenization:
                if use_word_embeddings or use_bpe_embeddings:
                    actual_inputs = [inputs, inputs2]
                else:
                    actual_inputs = inputs
            else:
                if use_word_embeddings or use_bpe_embeddings:
                    actual_inputs = inputs2
                else:
                    print 'No input features given!'
                    exit()

            model = Model(inputs=actual_inputs, outputs=predictions)

            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        if 'tag' not in params or ('tag' in params and not params['tag']):
            print model.summary()

        return model


    def get_intermediate_output_layer(self, layer_num, input, test_mode=True):
        """
        Get the intermediate output layer from a model
        :param model:
        :param layer_num:
        :param input:
        :param test_mode:
        :return:
        """
        params = self.parameters

        # Input params
        use_bytes = params['use_bytes']
        use_bpe = params['use_bpe']
        use_tokenization = params['use_tokenization']
        use_bpe_embeddings = params['use_bpe_embeddings']
        use_word_embeddings = params['use_word_embeddings']

        inputs = []
        if use_bytes or use_bpe or use_tokenization:
            assert('Input' in str(self.model.layers[0]))
            inputs.append(self.model.layers[0].input)
        if use_bpe_embeddings or use_word_embeddings:  # assume this is the second layer in the model
            assert('Input' in str(self.model.layers[0]))
            inputs.append(self.model.layers[1].input)

        get_x_layer_output = K.function(inputs + [K.learning_phase()], [self.model.layers[layer_num].output])
        if test_mode:
            layer_output = get_x_layer_output(input + [0])[0]
        else:
            layer_output = get_x_layer_output(input + [1])[0]
        return layer_output


    def crf_predict_proba(self, preds, crf_inputs):
        """
        Use CRF layer to predict probabilities of predicted sequences
        :param preds:
        :param crf_inputs:
        :return:
        """
        return self.model.layers[-1].predict_proba(K.variable(preds), K.variable(crf_inputs, dtype='int32'))

    def get_sample_vector(self, input):
        """
        Use the merging of different embedding features per sample as the representative vector of the sample
        :param input:
        :return:
        """
        return self.get_intermediate_output_layer(-2, input, test_mode=True)
