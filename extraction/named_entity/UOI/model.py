import keras
import keras.backend as K
from keras_wc_embd import get_embedding_layer


def _loss(layers, rnn_units, lmbd):
    """Generate loss function.

    :param layers: Parallel RNN layers.
    :param rnn_units: Unit size for each RNN.
    :param lmbd: A constant controlling the weights of losses.

    :return loss: The loss function.
    """

    def __loss(y_true, y_pred):
        kernel_cs_forward, kernel_cs_backward = [], []
        for (forward, backward) in layers:
            kernel_c_forward = forward.trainable_weights[1][:, rnn_units * 2:rnn_units * 3]
            kernel_c_backward = backward.trainable_weights[1][:, rnn_units * 2:rnn_units * 3]
            kernel_cs_forward.append(K.reshape(kernel_c_forward, (rnn_units * rnn_units,)))
            kernel_cs_backward.append(K.reshape(kernel_c_backward, (rnn_units * rnn_units,)))
        phi_forward = K.stack(kernel_cs_forward)
        phi_backward = K.stack(kernel_cs_backward)
        loss_sim_forward = K.sum(K.square(K.dot(phi_forward, K.transpose(phi_forward)) - K.eye(len(layers))))
        loss_sim_backward = K.sum(K.square(K.dot(phi_backward, K.transpose(phi_backward)) - K.eye(len(layers))))
        loss_cat = keras.losses.categorical_crossentropy(y_true, y_pred)
        return loss_cat + lmbd * (loss_sim_forward + loss_sim_backward)

    return __loss


def build_model(rnn_num,
                rnn_units,
                word_dict_len,
                char_dict_len,
                max_word_len,
                output_dim,
                word_dim=100,
                char_dim=100,
                char_embd_dim=100,
                lmbd=0.01,
                word_embd_weights=None):
    """Build model for NER.

    :param rnn_num: Number of parallel RNNs.
    :param rnn_units: Unit size for each RNN.
    :param word_dict_len: The number of words in the dictionary.
    :param char_dict_len: The numbers of characters in the dictionary.
    :param max_word_len: The maximum length of a word in the dictionary.
    :param output_dim: The output dimension / number of NER types.
    :param word_dim: The dimension of word embedding.
    :param char_dim: The final dimension of character embedding.
    :param char_embd_dim: The embedding dimension of characters before bidirectional RNN.
    :param lmbd: A constant controlling the weights of losses.
    :param word_embd_weights: Pre-trained embeddings for words.

    :return model: The built model.
    """
    inputs, embd_layer = get_embedding_layer(
        word_dict_len=word_dict_len,
        char_dict_len=char_dict_len,
        max_word_len=max_word_len,
        word_embd_dim=word_dim,
        char_hidden_dim=char_dim // 2,
        char_embd_dim=char_embd_dim,
        word_embd_weights=word_embd_weights,
    )
    rnns, layers = [], []
    for i in range(rnn_num):
        lstm_layer = keras.layers.LSTM(
            units=rnn_units,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True,
        )
        bi_lstm_layer = keras.layers.Bidirectional(
            layer=lstm_layer,
            name='LSTM_%d' % (i + 1),
        )
        layers.append((bi_lstm_layer.forward_layer, bi_lstm_layer.backward_layer))
        rnns.append(bi_lstm_layer(embd_layer))
    concat_layer = keras.layers.Concatenate(name='Concatenation')(rnns)
    dense_layer = keras.layers.Dense(units=output_dim, activation='softmax', name='Dense')(concat_layer)
    model = keras.models.Model(inputs=inputs, outputs=dense_layer)
    loss = _loss(layers, rnn_units, lmbd=lmbd)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=loss,
        metrics=[keras.metrics.categorical_accuracy],
    )
    return model
