# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, Embedding, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras import backend as K


def init_model(max_sequence_length, embedding_matrix, dropout, recurrent_dropout,
               vocab_size, lstm_hidden_layers=50, embedding_dim=300):
    # The input layer receives the sequence index of each word of input sentence
    left_input = Input(shape=(max_sequence_length,), dtype='int32')
    right_input = Input(shape=(max_sequence_length,), dtype='int32')

    # Word Embedding lookup Layer
    embedding_layer = Embedding(vocab_size, embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    left_encoder = embedding_layer(left_input)
    right_encoder = embedding_layer(right_input)

    # Recurrent Layer
    recurrent_layer = GRU(lstm_hidden_layers, implementation=2)
    #recurrent_layer = LSTM(lstm_hidden_layers, implementation=2)

    left_lstm = recurrent_layer(left_encoder)
    right_lstm = recurrent_layer(right_encoder)

    def out_shape(shapes):
        return (None, 1)

    def exponent_neg_manhattan_distance(vector):
        return K.exp(-K.sum(K.abs(vector[1] - vector[0]), axis=1, keepdims=True))

    malstm_distance = Lambda(exponent_neg_manhattan_distance, output_shape=out_shape)([left_lstm, right_lstm])
    
    return Model([left_input, right_input], [malstm_distance])
