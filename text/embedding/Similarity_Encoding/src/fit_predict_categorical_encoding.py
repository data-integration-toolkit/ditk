import os
import socket
import inspect
import itertools
import time
import datetime

import logging as logs
import numpy as np
from joblib import Parallel, delayed
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit

from fns_categorical_encoding import fit_predict_fold, verify_if_exists, \
    write_npy
from datasets import Data, get_data_folder
from constants import sample_seed, shuffle_seed, clf_seed, dropout
from model import NNetEstimator
# from model import NNetEstimator, NNetRegressor, NNetBinaryClassifier, \
#     NNetMultiClassifier

logs.basicConfig(level=logs.DEBUG)


def instanciate_estimators(clf_type, y=None, **kw):
    if clf_type in ['binary-clf']:
        print(('Fraction by class: True: %0.2f; False: %0.2f'
               % (list(y).count(True) / len(y),
                  list(y).count(False) / len(y))))
        cw = 'balanced'
        clfs = [
                # linear_model.LogisticRegressionCV(
                #     class_weight=cw, max_iter=100,
                #     penalty='l2', n_jobs=1),
                linear_model.RidgeClassifierCV(
                    class_weight=cw, cv=3),
                ensemble.GradientBoostingClassifier(
                   n_estimators=100),
                # ensemble.RandomForestClassifier(
                #     n_estimators=100, class_weight=cw)
                # neural_network.MLPClassifier(
                #     hidden_layer_sizes=(100,)),
                # NNetBinaryClassifier(**kw)
                # waiting for data preprocessing to get configs
                ]

    elif clf_type in ['multiclass-clf']:
        print('fraction of the most frequent class:',
              max([list(y).count(x)
                   for x in set(list(y))]) / len(list(y)))
        clfs = [
                # linear_model.LogisticRegressionCV(
                #     max_iter=100, penalty='l2', n_jobs=1),
                linear_model.RidgeClassifierCV(cv=3),
                ensemble.GradientBoostingClassifier(
                    n_estimators=100),
                # ensemble.RandomForestClassifier(
                #     n_estimators=100),
                # neural_network.MLPClassifier(hidden_layer_sizes=(100,)),
                # NNetMultiClassifier(**kw)
                ]
    elif clf_type in ['regression']:
        clfs = [
                linear_model.RidgeCV(cv=3),
                ensemble.GradientBoostingRegressor(
                    n_estimators=100),
                ensemble.RandomForestRegressor(
                    n_estimators=100)
                # neural_network.MLPRegressor(hidden_layer_sizes=(100,))
                # NNetRegressor(**kw)
                # waiting for data preprocessing to get configs
                ]
    else:
        raise ValueError("{} not recognized".format(clf_type))
    return clfs


def select_shuffle_split(clf_type, n_splits, test_size):
    if clf_type in ['regression', 'multiclass-clf']:
        ss = ShuffleSplit(n_splits=n_splits,
                          test_size=test_size,
                          random_state=shuffle_seed)
    else:
        ss = StratifiedShuffleSplit(n_splits=n_splits,
                                    test_size=test_size,
                                    random_state=shuffle_seed)
    return ss


def choose_nrows(dataset_name):
    if dataset_name in ['docs_payments', 'crime_data',
                        'traffic_violations', 'federal_election',
                        'public_procurement']:
        n_rows = 100000  # -1 if using all rows for prediction
    elif dataset_name in ['beer_reviews', 'road_safety']:
        n_rows = 10000
    else:
        n_rows = -1
    return n_rows


def fit_predict_categorical_encoding(datasets, n_jobs, n_splits, test_size,
                                     encoders, str_preprocess,
                                     dimension_reductions, results_path,
                                     model_path=None):
    '''
    Learning with dirty categorical variables.
    '''
    logger = logs.getLogger('{},{}'.format(
        __name__, inspect.currentframe().f_code.co_name))
    path = get_data_folder()
    results_path = os.path.join(path, results_path)
    model_path = os.path.join(path, model_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for dataset in datasets:
        n_rows = choose_nrows(dataset_name=dataset)
        for encoder in encoders:
            logger.debug('Dataset:{}'.format(dataset))
            data = Data(dataset).get_df()
            data.preprocess(n_rows=n_rows, str_preprocess=str_preprocess)
            for dimension_reduction in dimension_reductions:
                logger.debug('Data shape: %d, %d' % data.df.shape)
                ss = select_shuffle_split(data.clf_type, n_splits, test_size)
                scaler = preprocessing.StandardScaler(with_mean=False)

                # Define classifiers
                clfs = instanciate_estimators(
                    data.clf_type,
                    y=data.df.loc[:, data.ycol].values,
                    model_path=model_path, dropout=dropout)

                for clf in clfs:
                    logger.info(
                        '{}: {} \n{}: {} \n{}: {} \n{}: {} \n{}: {},{}'.format(
                            'Prediction column', data.ycol,
                            'Task', str(data.clf_type),
                            'Classifier', clf,
                            'Encoder', encoder,
                            'Dimension reduction', dimension_reduction[0],
                            dimension_reduction[1]))

                    if not isinstance(clf, NNetEstimator):
                        if 'random_state' in clf.get_params():
                            clf.set_params(random_state=clf_seed)
                    results_dict = {'dataset': data.name,
                                    'n_splits': n_splits,
                                    'test_size': test_size,
                                    'n_rows': n_rows,
                                    'encoder': encoder,
                                    'str_preprocess': str_preprocess,
                                    'clf': [clf.__class__.__name__,
                                            clf.get_params()],
                                    'ShuffleSplit':
                                        [ss.__class__.__name__],
                                    'scaler': [scaler.__class__.__name__,
                                               scaler.get_params()],
                                    'sample_seed': sample_seed,
                                    'shuffleseed': shuffle_seed,
                                    'col_action': data.col_action,
                                    'clf_type': data.clf_type,
                                    'dimension_reduction':
                                        dimension_reduction
                                    }
                    # if verify_if_exists(results_path, results_dict):
                    #     print('Prediction already exists.\n')
                    #     continue
                    start = time.time()
                    MX, y = (data.df.loc[:, data.xcols].values,
                             data.df.loc[:, data.ycol].values)
                    data.make_configs(encoder=encoder)
                    #print("====================================================================")
                    #print(MX)
                    #print()
                    #print(data.df[['Employee Position Title']])
                    #print("====================================================================")
                    pred = Parallel(n_jobs=n_jobs)(
                        delayed(fit_predict_fold)(
                            MX, y, train_index, test_index,
                            data.col_action, data.xcols, data.name, encoder,
                            fold, n_splits, clf, data.clf_type, scaler,
                            dimension_reduction, configs=data.configs)
                        for (train_index, test_index), fold
                        in zip(ss.split(MX, y), range(1, n_splits + 1)))
                    pred = list(itertools.chain.from_iterable(pred))
                    pred = np.array(pred)
                    results = {'fold': list(pred[:, 0]),
                               'n_train_samples': list(pred[:, 1]),
                               'n_train_features': list(pred[:, 2]),
                               'score': list(pred[:, 3]),
                               'encoding_time': list(pred[:, 4]),
                               'training_time': list(pred[:, 5])}
                    results_dict['results'] = results

                    # Saving results
                    pc_name = socket.gethostname()
                    now = ''.join([c for c in str(datetime.datetime.now())
                                   if c.isdigit()])
                    results_file = os.path.join(
                        results_path, pc_name + '_' + now + '.npy')

                    write_npy(results_dict, results_file)
                    print('prediction time: %.1f s.' % (time.time() - start))
                    print('Saving results to: %s\n' % results_file)
