import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from re import sub, findall
from math import ceil
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod

from keras.optimizers import adam
from keras.utils import plot_model, to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda, Dropout
from constants import common_network_shape, batch_size
from keras.metrics import mse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from keras.metrics import binary_crossentropy, mean_squared_error, categorical_crossentropy

from constants import val_split, epochs, LOSS_MAP, rescaling_factor, tf_allowed_scope


def _get_min_max_idxs(shapes):
    cum_len = np.cumsum(shapes)
    min_idxs = np.insert(cum_len[:-1], 0, [0])
    return list(zip(min_idxs, cum_len))


# from Data import Data

class Config:
    """
    input configuration for neural network
    """

    def __init__(self, name=None, kind=None, distance=None, encoder=None,
                 shape=None, encoding_layer_dims=(3, 3, 3),
                 clf=None):
        """
        :param name: name of the column
        :param encoder:
        :param shape:
        :param encoding_layer_dims:
        :param clf:
        """
        self.distance = distance
        self.clf = clf
        self.name = name
        self.encoder = encoder
        self.shape = shape
        self.kind = kind
        self.nn_kwargs = {'encoding_layer_dims': encoding_layer_dims,
                          'depth': len(encoding_layer_dims)}

    def rescale_layer_number(self, factor=1 / 5):
        # using ceil because this rescaling might lead to 0-shape
        self.nn_kwargs['encoding_layer_dims'] = tuple(
            [ceil(self.shape[-1] * factor)
             for _ in range(self.nn_kwargs['depth'])])

    def __repr__(self):
        return 'Config for feature {}'.format(self.name)


class NNetEstimator:
    __metaclass__ = ABCMeta
    """
    benchmark neural network model
    """

    def __init__(self, model_path=None, dropout=None):
        self.configs = None
        self.common_network_shape = list(common_network_shape)
        self.model_path = model_path
        self.model = None
        self.common_activation = None
        self.metrics = None
        self.dropout = dropout

    def build(self, configs, output_dim=1):
        self.configs = configs
        shapes = [c.shape[-1] for c in configs]
        X_tot = Input(shape=(sum(shapes),))
        if self.dropout is not None:
            dropout_layer = Dropout(rate=self.dropout)
            x_tot_dropout = dropout_layer(X_tot)
        else:
            x_tot_dropout = X_tot
        xs_encoding_pipeline, encoding_layers = self.make_encoding_network(x_tot_dropout, shapes)

        concat_layer = Concatenate(axis=1)
        common_x = concat_layer([inp[-1] for inp in xs_encoding_pipeline])

        common_x_pipeline, common_layers = self.make_common_network(common_x, output_dim)
        y = common_x_pipeline[-1]
        self.model = Model(X_tot, y)
        # compile model
        opt = adam()
        self.model.compile(optimizer=opt, loss=self.metrics, metrics=[self.metrics])
        self.plot_model_architecture(name='model.png')

    def fit(self, X, y, configs, draw_learning_curve=True, **kwargs):
        self.build(configs, output_dim=y.shape[-1])
        summary = self.model.fit(X, y, epochs=epochs, validation_split=val_split, **kwargs)
        self.draw_learning_curve(summary, save=True, name='summary.png')

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def make_common_network(self, x_input, output_dim):
        common_layers, common_x_pipeline = [], [x_input]
        self.common_network_shape[-1] = output_dim
        for i, n_units in enumerate(self.common_network_shape):
            common_layers.append(Dense(n_units, activation=self.common_activation, name='common_dense_{}'.format(i)))
            common_x_pipeline.append(common_layers[-1](common_x_pipeline[-1]))
        return common_x_pipeline, common_layers

    def plot_model_architecture(self, name):
        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            plot_model(self.model, to_file=os.path.join(self.model_path, name))
        else:
            raise AttributeError('model does not have a model_path')

    def make_encoding_network(self, x_input: Input, shapes):
        xs_in, xs_outs, layers = [], [], []
        idxs = _get_min_max_idxs(shapes)
        for i, c in enumerate(self.configs):
            x_pipeline, layer_pipeline = self._make_encoding_layer_feature(x_input, idxs[i][0], idxs[i][1], c)
            layers.append(layer_pipeline)
            xs_in.append(x_pipeline)
        return xs_in, layers

    def _make_encoding_layer_feature(self, x_input, min_idx, max_idx, config):
        x_in_pipeline, layer_pipeline = [], []
        feature, split_layer = self._make_feature_selection_layer(x_input, min_idx, max_idx, config)
        layer_pipeline.append(split_layer), x_in_pipeline.append(feature)
        for i, n_units in enumerate(config.nn_kwargs['encoding_layer_dims']):
            layer_name = '_'.join(findall(tf_allowed_scope, config.name))  # filter non allowed characters
            layer_pipeline.append(
                Dense(n_units, input_shape=config.shape, name='encoding_{}_{}'.format(i, layer_name)))
            x_in_pipeline.append(layer_pipeline[-1](x_in_pipeline[-1]))
        return x_in_pipeline, layer_pipeline

    def _make_feature_selection_layer(self, x_input, min_idx, max_idx, config: Config):
        layer_name = '_'.join(findall(tf_allowed_scope, config.name))  # filter non-allowed characters
        splitting_layer = Lambda(lambda x: x[:, min_idx:max_idx],
                                 name='{}_selection'.format(layer_name))
        feature = splitting_layer(x_input)
        return feature, splitting_layer

    def set_params(self, *args, **kwargs):
        pass

    def get_params(self, *args, **kwargs):
        pass

    def draw_learning_curve(self, summary, save=True, name=None):
        """
        for overfitting/underfitting issues
        :param summary: keras.summary object that gathers metrics at each epoch/
        :param save: if True, saves the figure
        :param path: path to save the figure
        :return:
        """
        metric_name = self.metrics.__name__
        df = pd.DataFrame({metric_name: summary.history[metric_name],
                           'val_{}'.format(metric_name): summary.history['val_{}'.format(metric_name)]})
        f, ax = plt.subplots()
        df.plot(ax=ax)
        if save and (name is not None):
            f.savefig(os.path.join(self.model_path, name))
        plt.close('all')


class NNetRegressor(NNetEstimator):
    def __init__(self, model_path=None, dropout=None):
        super().__init__(model_path=model_path, dropout=dropout)
        self.metrics = mean_squared_error
        self.common_activation = 'relu'

    def fit(self, X, y, configs, draw_learning_curve=True, **kwargs):
        super().fit(X, y, configs, draw_learning_curve=draw_learning_curve, **kwargs)

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)


class NNetBinaryClassifier(NNetEstimator):
    def __init__(self, model_path=None, dropout=None):
        super(NNetBinaryClassifier, self).__init__(model_path=model_path, dropout=dropout)
        self.y_label_encoder = LabelEncoder()
        self.metrics = binary_crossentropy
        self.common_activation = 'sigmoid'
        self.precomputed_y = False

    def fit(self, X, y, configs, draw_learning_curve=True, **kwargs):
        if y.dtype == 'int':
            y_int = y
            self.precomputed_y = True
        else:
            y_int = self.y_label_encoder.fit_transform(y).reshape((-1, 1))
        assert np.mean(y) < 0.5, 'the least frequent class should be 1. This is not the case with this dataset'
        super().fit(X, y_int, configs, draw_learning_curve=draw_learning_curve, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return super().predict(*args, **kwargs).reshape((-1, 1))

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs).reshape((-1, 1))


class NNetMultiClassifier(NNetEstimator):
    def __init__(self, model_path=None, dropout=None):
        super().__init__(model_path=model_path, dropout=dropout)
        self.y_label_encoder = LabelBinarizer()
        self.metrics = categorical_crossentropy
        self.common_activation = 'sigmoid'

    def fit(self, X, y, configs, draw_learning_curve=True, **kwargs):
        y_int = self.y_label_encoder.fit_transform(y)
        super().fit(X, y_int, configs, draw_learning_curve=draw_learning_curve, **kwargs)

    def predict(self, *args, **kwargs):
        return self.y_label_encoder.inverse_transform(super().predict(*args, **kwargs))
