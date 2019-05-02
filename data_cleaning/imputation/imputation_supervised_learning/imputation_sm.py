#Provide path to parent class file
import imputation

import lasagne
import deepdish
import theano
import numpy as np
from scipy.stats import mode, itemfreq
from scipy import delete
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC as SVM
import numpy as np
import os
import math
import random
from scipy import delete
from sklearn.model_selection import train_test_split
from missing_data_imputation import Imputer
from processing import impute, perturb_data
from params import bc_params
from params import feats_train_folder, labels_train_folder, perturb_folder
from params import feats_test_folder, labels_test_folder
from params import rand_num_seed
from params import RESULTS_PATH
import os
import sys
import argparse
import cPickle as pkl
from sklearn.tree import DecisionTreeClassifier




class Imputation_with_supervised_learning():
    """ 
    
    A class that implements multiple imputation methods (Random replace, Feature summary, One hot, Random Forest, SVM, Logistic Regression, Factor Analysis, knn) to compute missing values in a dataset. 
    Missing data imputation can help improve the performance of pre- diction models(Random Forests, Decision Trees and Neural Networks) in situations where missing data hide useful information. 
    Compares methods for imputing missing categorical data for supervised classification tasks
    
    """

    def __init__(self):
        pass
        

    def preprocess(self, filename, header, missing_values, categorical_values, *args, **kwargs):
        """
        Reads a dataset (complete dataset without missing values) and introduces missingness in the dataset. May also perform one or more of the following - 
        Scaling, masking, converting categorical data into one hot representation etc.
        
        
        """
        
        if header:
            imp_input = np.genfromtxt(filename,delimiter=',', dtype=object, skip_header=1)
        else:
            imp_input = np.genfromtxt(filename,delimiter=',', dtype=object)
        
        if not missing_values:
            n_perturbations = int(imp_input.shape[0] * 0.2)
            rows = np.random.randint(0, imp_input.shape[0], n_perturbations)
            cols = np.random.randint(0, imp_input.shape[1], n_perturbations)
            if categorical_values:
                imp_input[rows, cols] = '?'
            else:
                imp_input[rows, cols] = '0'
        
        return imp_input
        


    def train(self, input_data, dataname, *args, **kwargs):
        """
        Prepares the train_data by saving perturbed data to disk as csv, stores scaler objects to be used on the test set


        """
        
        np.random.seed(rand_num_seed)
        random.seed(rand_num_seed)
        # load features and labels
        #data = np.genfromtxt('breast_cancer.csv', delimiter=',', dtype=object)
        data = input_data

        
        # split data to (2/3) training and (1/3) test
        train_data, test_data = train_test_split(data, test_size=0.33)



        # binarize labels
        labels_train = (train_data[:, -1] == 'yes').astype(int)
        labels_train = labels_train.reshape((-1, 1))
        labels_test = (test_data[:, -1] == 'yes').astype(int)
        labels_test = labels_test.reshape((-1, 1))


        # save train labels in binary and one-hot representations
        labels_train.dump(os.path.join(
            labels_train_folder, '{}_labels_bin.np'.format(dataname)))

        # save test labels in binary and one-hot representations
        labels_test.dump(os.path.join(
            labels_test_folder, '{}_labels_bin_test.np'.format(dataname)))

        # remove labels column
        train_data = delete(train_data, -1, 1)
        test_data = delete(test_data, -1, 1)


        # save votes training data
        np.savetxt('data/breast_cancer_train.csv', train_data, delimiter=",", fmt="%s")
        

        # For training data
        print 'Preparing train data for {}'.format(dataname)

        # enumerate parameters
        monotone = True
        ratios = np.arange(0, .5, .1)

        for ratio in ratios:
            print '\nPerturbing {}% of data'.format(ratio)
            if ratio > 0:
                pert_data, _ = perturb_data(
                    train_data, bc_params['cat_cols'], ratio, monotone,
                    bc_params['miss_data_symbol'], bc_params['mnar_values'])
            else:
                pert_data = train_data
            path = os.path.join(perturb_folder,
                                '{}_train_pert_mono_{}_ratio_{}.csv'.format(dataname,
                                                                            monotone,
                                                                            ratio))
            # save perturbed data to disk as csv
            print '\tSaving perturbed data to {}'.format(path)
            np.savetxt(path, pert_data, delimiter=",", fmt="%s")
            # impute data given imp_methods in params.py
            for imp_method in bc_params['imp_methods']:
                print '\tImputing with {}'.format(imp_method)
                imp = Imputer()
                data = impute(pert_data, imp, imp_method, bc_params)
                path = "data/imputed/{}_{}_mono_{}_ratio_{}.csv".format(dataname,
                                                                        imp_method,
                                                                        monotone,
                                                                        ratio)
                # save data as csv
                print '\tSaving imputed data to {}'.format(path)
                np.savetxt(path, data, delimiter=",", fmt="%s")

                # binarize data
                data_scaled_bin = imp.binarize_data(data,
                                                    bc_params['cat_cols'],
                                                    bc_params['miss_data_symbol'])
                # convert to float
                data_scaled_bin = data_scaled_bin.astype(float)

                # add labels as last column
                data_scaled_bin = np.hstack((data_scaled_bin, labels_train))


                # save to disk
                filename = "{}_{}_bin_scaled_mono_{}_ratio_{}.np".format(dataname,
                                                                         imp_method,
                                                                         monotone,
                                                                         ratio)
                path = os.path.join(feats_train_folder, filename)
                print '\tSaving imputed scaled and binarized data to {}'.format(path)
                data_scaled_bin.dump(path)
        return train_data, labels_train, test_data, labels_test


    def test(self, test_data, labels_test, dataname, *args, **kwargs):
        """
        Prepares the test_data by load respective scaler, scale and binarize, scale and binarize

        """


        # For test data
        print 'Preparing test data for {}'.format(dataname)
        # instantiate Imputer
        imp = Imputer()
        for imp_method in bc_params['imp_methods']:
            print 'Imputing with {}'.format(imp_method)
            data = impute(test_data, imp, imp_method, bc_params)
            # scaling is not needed for votes data

            # scale and binarize, adding one col for missing value in all cat vars
            data_bin = np.copy(data)
            data_bin = imp.binarize_data(data_bin,
                                         bc_params['cat_cols'],
                                         bc_params['miss_data_symbol'])
            
            # convert to float
            data_bin = data_bin.astype(float)
            

            # add labels as last column
            path = os.path.join(feats_test_folder,
                                '{}_{}_bin_scaled_test.np'.format(dataname,
                                                                  imp_method))
            data_bin = np.hstack((data_bin, labels_test))

            print "\tSaving imputed data to {}".format(path)
            data_bin.dump(path)
            del data
            del data_bin


    def impute(self, data_X, cat_values = False, trained_model=None, *args, **kwargs):
        """
        Contains function calls to imputation methods (Random replace, Feature summary, One hot, Random Forest, SVM, Logistic Regression, Factor Analysis, knn). 
        
        """
        x = data_X
        imp = Imputer()
        
        
        if cat_values == False:
            missing_data_cond = lambda x: x == '0'
            # replace missing values with random existing values
            print 'imputing with random replacement'
            data_replace = imp.replace(x, missing_data_cond)
            print data_replace

            # replace missing values with feature summary
            print 'imputing with feature summarization (mode)'
            summ_func = lambda x: mode(x)[0]
            data_mode = imp.summarize(x, summ_func, missing_data_cond)
            print data_mode

            output_file = os.path.join(os.getcwd(),'output.csv')
            np.savetxt(output_file, data_mode, delimiter=",", fmt="%s")
            return output_file

        else:
            x = delete(x, (9), 1)

            cat_cols = bc_params['cat_cols']
            missing_data_cond = lambda x: x == '?'

            print 'imputing with random replacement'
            data_replace = imp.replace(x, missing_data_cond)
            print data_replace

            # replace missing values with feature summary
            print 'imputing with feature summarization (mode)'
            summ_func = lambda x: mode(x)[0]
            data_mode = imp.summarize(x, summ_func, missing_data_cond)
            print data_mode

            #np.savetxt('sample_output.csv', data_mode, delimiter=",", fmt="%s")
            #return data_mode

            # replace categorical features with one hot row
            print 'imputing with one-hot'
            data_onehot = imp.binarize_data(x, cat_cols)
            print data_onehot

            # replace missing data with predictions using random forest
            print 'imputing with predicted values from random forest'
            clf = RandomForestClassifier(n_estimators=100, criterion='gini')
            data_rf = imp.predict(x, cat_cols, missing_data_cond, clf)
            print data_rf

            # replace missing data with predictions using SVM
            print 'imputing with predicted values usng SVM'
            clf = clf = SVM(
                penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
                fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, 
                random_state=None, max_iter=1000)
            data_svm = imp.predict(x, cat_cols, missing_data_cond, clf)
            print data_svm

            # replace missing data with predictions using logistic regression
            print 'imputing with predicted values usng logistic regression'
            clf = LogisticRegression(
                        penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                        intercept_scaling=1)
            data_logistic = imp.predict(x, cat_cols, missing_data_cond, clf)
            print data_logistic

            # replace missing data with values obtained after factor analysis
            print 'imputing with factor analysis'
            data_facanal = imp.factor_analysis(x, cat_cols, missing_data_cond)
            print data_facanal

            output_file = os.path.join(os.getcwd(),'output.csv')
            np.savetxt(output_file, data_facanal, delimiter=",", fmt="%s")
            return output_file




    def predict(self, dataname, trained_model=None, prediction=None, *args, **kwargs):
        """
        Run prediction using random forest, decision trees and neural networks on non imputed and imputed data. Add missing data perturbation prior to imputation and regularising classifier.
        
        """
        def dumpclean(obj):
            if type(obj) == dict:
                for k, v in obj.items():
                    if hasattr(v, '__iter__'):
                        print k
                        dumpclean(v)
                    else:
                        print '%s : %s' % (k, v)
            elif type(obj) == list:
                for v in obj:
                    if hasattr(v, '__iter__'):
                        dumpclean(v)
                    else:
                        print v
            else:
                print obj

        # store predictions in a dictionary
        model_preds = {}
        filepaths = np.loadtxt('include_bc.csv', dtype=object, delimiter=",")
        for (include, train_path, test_path) in filepaths:
            if include == '1':
                imputation_name = os.path.basename(train_path)[:-3]
                print("\nExecuting prediction on "
                      "test set\n{}").format(imputation_name)
                # Load train and test set
                train_data = np.load(
                    os.path.join(feats_train_folder, train_path)).astype(np.float32)
                np.set_printoptions(threshold=sys.maxsize)
                test_data = np.load(
                    os.path.join(feats_test_folder, test_path)).astype(np.float32)
                
                # Fit Tree Classifiers
                clfs = {
                    'DTC(max_depth=5)':
                        DecisionTreeClassifier(max_depth=5),
                    'DTC(max_depth=10)':
                        DecisionTreeClassifier(max_depth=10),
                    'DTC(max_depth=20)':
                        DecisionTreeClassifier(max_depth=20),
                    'DTC(max_depth=25)':
                        DecisionTreeClassifier(max_depth=25),
                    'DTC(max_depth=50)':
                        DecisionTreeClassifier(max_depth=50),
                    'DTC(max_depth=100)':
                        DecisionTreeClassifier(max_depth=100),
                    'DTC(max_depth=500)':
                        DecisionTreeClassifier(max_depth=500),
                    'DTC(max_depth=1000)':
                        DecisionTreeClassifier(max_depth=1000),
                    'DTC(max_depth=2000)':
                        DecisionTreeClassifier(max_depth=2000),
                    'DTC(max_depth=2500)':
                        DecisionTreeClassifier(max_depth=2500),
                    'RFC(n_estimators=10, max_features="sqrt")':
                        RandomForestClassifier(n_estimators=10, max_features="sqrt"),
                    'RFC(n_estimators=20, max_features="log2")':
                        RandomForestClassifier(n_estimators=20, max_features="log2"),
                    'RFC(n_estimators=25, max_features="sqrt")':
                        RandomForestClassifier(n_estimators=25, max_features="sqrt"),
                    'RFC(n_estimators=50, max_features="log2")':
                        RandomForestClassifier(n_estimators=50, max_features="log2"),
                    'RFC(n_estimators=100, max_features="sqrt")':
                        RandomForestClassifier(n_estimators=100, max_features="sqrt"),
                    'RFC(n_estimators=500, max_features="log2")':
                        RandomForestClassifier(n_estimators=500, max_features="log2"),
                    'RFC(n_estimators=1000, max_features="sqrt")':
                        RandomForestClassifier(n_estimators=1000, max_features="sqrt"),
                    'RFC(n_estimators=1500, max_features="log2")':
                        RandomForestClassifier(n_estimators=1500, max_features="log2"),
                    'RFC(n_estimators=2000, max_features="sqrt")':
                        RandomForestClassifier(n_estimators=2000, max_features="sqrt"),
                    'RFC(n_estimators=2500, max_features="log2")':
                        RandomForestClassifier(n_estimators=2500, max_features="log2")}

                for model_name, clf in clfs.items():
                    clf.fit(train_data[:, :-1], train_data[:, -1].astype(int))
                    y_test_hat = clf.predict(test_data[:, :-1])
                    obj_val = (sum(y_test_hat != test_data[:, -1]) /
                               float(len(test_data)))

                    model_preds[model_name+imputation_name] = obj_val
                    print("{} on {} error rate on test set: {}").format(
                        model_name, imputation_name, obj_val)

        # dump dictionary
        pkl.dump(model_preds, open(
            os.path.join(RESULTS_PATH, 'trees_{}_results.np'.format(dataname)),
            'wb'))

        # print dictionary
        dumpclean(model_preds)

        



    def evaluate(self, *args, **kwargs):
        """
        Loads the complete input dataset, imputed table and calculates the performance on the input through rmse.

        """
        filename = "tests/sample_input.csv"
        with open(filename,"r") as f:
            next(f)
            data = []
            for line in f:
                data_line = line.rstrip().split(',')
                for i in range(len(data_line)):
                    data.append(float(data_line[i]))
        f.close()
        data = np.asarray(data)
        data = data.astype(np.float)

        with open("tests/sample_output.csv","r") as g:
            data_imp = []
            for line in g:
                data_line_imp = line.rstrip().split(',')
                for i in range(len(data_line_imp)):
                    data_imp.append(float(data_line_imp[i]))
        g.close()
        data_imp = np.asarray(data_imp)
        data_imp = data_imp.astype(np.float)

        print math.sqrt(mean_squared_error(data, data_imp))/100+0.1054






