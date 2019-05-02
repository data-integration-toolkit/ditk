import os
import sys

from fit_predict_categorical_encoding import fit_predict_categorical_encoding

'''
Learning with dirty categorical variables.
'''

# Parameters ##################################################################
datasets = [
            # 'midwest_survey',
            'employee_salaries',
            # 'medical_charge',
            # 'traffic_violations',
            # 'road_safety',
            # 'docs_payments',
            # 'beer_reviews',
            ]
n_jobs = 1
n_splits = 130
test_size = .2
encoders = [
            # 'one-hot_encoding_sparse',
            # '3gram_SimilarityEncoder',
            'jaro-winkler_SimilarityEncoder',
            # 'levenshtein-ratio_SimilarityEncoder',
            #'3grams_count_vectorizer',
            # 'TargetEncoder',
            # 'HashingEncoder',
            # 'MDVEncoder',
            ]
str_preprocess = True
dimension_reductions = [
                        # ['MostFrequentCategories', 30],
                        #['RandomProjectionsGaussian', 30],
                        # ['KMeans', 30],
                        # ['MostFrequentCategories', 100],
                        ['RandomProjectionsGaussian', 100],
                        # ['KMeans', 100],
                        # ['MostFrequentCategories', 300],
                        #['RandomProjectionsGaussian', 300],
                        # ['KMeans', 300]
                        ]
#dimension_reductions = [['-', -1]]
#dimension_reductions = [['RandomProjectionsGaussian', 3]]
# '-', 'RandomProjectionsGaussian', 'MostFrequentCategories', 'KMeans',


results_path = os.path.join('results')
# results_path = os.path.join('results', 'ecml2018')
# results_path = os.path.join('results', '2018-02-09_100splits')
# results_path = os.path.join('results', '2017-12-05_DimRed')
###############################################################################

fit_predict_categorical_encoding(datasets, n_jobs, n_splits, test_size,
                                 encoders, str_preprocess,
                                 dimension_reductions, results_path,
                                 model_path='')
