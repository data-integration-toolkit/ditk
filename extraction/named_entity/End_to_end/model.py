# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, TimeDistributed, Dropout, concatenate, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.initializers import RandomUniform
from keras.layers.embeddings import Embedding
import configs
import numpy as np
from keras_contrib.layers import CRF

def get_model(word_embeddings, max_length, chars_length, no_of_classes, weights=False):
    #character level embeddings
    init_value = np.sqrt(3/configs.CHAR_EMBDS_DIM)
    chars_input = Input(shape=(max_length,configs.MAX_CHARS,), dtype='int32', name='char_input')
    chars_emb = Embedding(input_dim = chars_length, output_dim = configs.CHAR_EMBDS_DIM,
                        embeddings_initializer=RandomUniform(minval=-init_value, maxval=init_value), trainable=True, name='char_emb')(chars_input)
    chars_cnn = TimeDistributed(Conv1D(kernel_size=configs.FILTER_SIZE, filters=configs.NO_OF_FILTERS, padding='same',activation='tanh', strides=1))(chars_emb) 
    max_out = TimeDistributed(MaxPooling1D(pool_size=configs.POOL_SIZE))(chars_cnn) 
    chars = TimeDistributed(Flatten())(max_out)
    chars = Dropout(configs.DROPOUT)(chars)
    
    # Word Embeddings
    words_input = Input(shape=(max_length,),dtype='int32',name='word_input')
    word_embed = Embedding(input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1],
                       weights=[word_embeddings], trainable=False, name='word_embed')(words_input)
    word_embed = Dropout(configs.DROPOUT)(word_embed)

    output = concatenate([word_embed,chars])
    output = Bidirectional(LSTM(max_length, return_sequences=True))(output)
    output = Dropout(configs.DROPOUT)(output)
    crf =  CRF(no_of_classes, sparse_target = True)
    output = crf(output)
    model = Model(inputs=[words_input, chars_input], outputs=[output])
    if weights:
        model.load_weights(configs.MODEL_FILE)
    model.compile(optimizer = 'adam',loss= crf.loss_function, metrics = [crf.accuracy])
    return model
   

