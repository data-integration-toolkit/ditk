# -*- coding: utf-8 -*-
"""
Module for transfer learning for named entity recognition (NER).

Please note that this is only for the fully supervised case, where both the
source and target domains have labeled data.

For neural network approaches, see train_bilstm_model.py and load_pretrained.py

Transfer learning methods implemented here:
    - src
    - tgt
    - all
    - augment: Daume's EasyAdapt
    - PRED
    - PRED-CCA

Usage
-----

# quickstart will load some data for demo purposes.
# >>> import quickstart as q

# To train a model on some source corpus and evaluate it on a target corpus:
# >>> import transferlearning as tl
# >>> D = tl.DomainAdaptation()
# >>> D.train('augment', 'CRF', q.src, q.tgt_train, q.tgt_test)
# >>> score = D.evaluate(q.tgt_test)
# >>> score.F1()

# To load a pre-trained model and evaluate it on a corpus:
# >>> import transferlearning as tl
# >>> D = tl.DomainAdaptation()
# >>> D.load_pretrained_model(modelname='pretrained-StanfordNER')
# >>> score = D.evaluate(q.src)
# >>> score.F1()

"""

import os
from nltk.tag.sequential import ClassifierBasedTagger
from nltk.classify import NaiveBayesClassifier, MaxentClassifier

from features import (ZhangJohnson,
                      wordembeddings_as_features,
                      combine_several_featfunctions)

from embedding_utils import (get_word_embeddings,
                             get_unique_words)
from label_mismatch import cca
import sentence_utils
from evaluation import Evaluator
from classifiers.averaged_perceptron import PerceptronNER
from classifiers.crf import CRFTagger
import utils
from stratified_split import writefile

# NLTK DATASETS & TOOLS
##########################
from nltk.data import path as nltk_data_path

TRAINED_MODEL_NAME = 'trained_model'

MODEL_PARAMS_NAME = 'model_params'

INSTANCE_VAR_NAME = 'instance_vars'
nltk_data_location = os.getenv('NLTK_DATA_PATH')
if nltk_data_location is not None:
    nltk_data_path.append(nltk_data_location)


# WORD EMBEDDINGS
##########################
embeddingsPath = 'word_embeddings/glove.6B.100d.txt'

class DomainAdaptation():
    """ This will make it easy to compare different NER classifiers, domain,
    adaptation techniques and features. One can train the model on a corpus
    or use a pre-trained model, and test the method on a labeled dataset.

    """
    def __init__(self, verbose=False):
        self._verbose = verbose
        self.pretrained_model = 'None' # This is changed to something else if using a pretrained model.
        self.path_addons = ''
        self.model = {}
        self.main_transfer_method = 'None'
        self.main_classifier_name = 'None'

        self.src_features = None
        self.tgt_features = None
        self.src_class_algo = None
        self.tgt_class_algo = None

        self.src_train = None
        self.tgt_train = None
        self.tgt_test = None
        self.model_params = None
        self.instance_variables = None
        self.excluded_entities = None
        self.dataset = None
        self.label2vec = None
        self.dataset = "DITK"

    def load_data(self, src_train, tgt_train, tgt_test, excluded_entities=None):


        if src_train:
            self.src_train = src_train
            self.model['entity_types_src'] = sentence_utils.get_tagset(self.src_train, with_prefix=False)


        self.tgt_train = tgt_train
        self.tgt_test = tgt_test
        self.model['entity_types_tgt'] = sentence_utils.get_tagset(self.tgt_test + self.tgt_train, with_prefix=False)

        self.excluded_entities = excluded_entities

    def set_parameters(self, transfer_method, classifier_name):
        self.main_transfer_method = transfer_method
        self.main_classifier_name = classifier_name


    def train(self, transfer_method, classifier_name, features_name='ZhangJohnson',
              **kwargs):
        """ Train the model with a given classifier and with a given domain
        adaptation method (preprocessing or post-processing).

        Parameters
        ----------
        transfer_method : str
            The name of the transfer method to use. They can be:
                * src: Train the model with source training data
                * tgt: Train the model with target training data
                * all: Train the model with both source and target data
                * augment:
                    Train the model both source and target data, but
                    enlarge the feature space, using Daume's easyadapt
                    method[1], so if a token i is in the source,
                    use feature (x_i, x_i, 0) instead of x_i for each feature;
                    if the token in the target data use feature (x_i, 0, x_i).
                    The first entry of the tuple stands for 'general' features,
                    the second is 'source only', and the third is 'target only'.
                 * pred: The 'PRED' method, described in Daume (#TODO put other
                    references in here).
                    Unlike the other methods, the train function both trains
                    and tests, saves the result. Calling 'test' merely prints
                    the score.
                    This permits another optional keyword argument, 'no_prefix':
                    if True, it removes the 'I-' or 'B-' from the PRED feature.


        classifier_name : str
            The name of the classifier to use. Roughly in order of performance:
                * CRF: the CRFTagger from nltk, which calls external CRFSuite.
				  Optional keyword parameter: 'algorithm', which can be either
				  'l2sgd' or 'lbfgs'. If not given, 'l2sgd' is used.
                * averaged_perceptron : the averaged perceptron from nltk
                * megam: nltk's binding from of Daume's external megam program
                * IIS: Improved Iterative Scaling, via nltk
                * GIS: Generalized Iterative Scaling, via nltk
                * naivebayes: Naive Bayes from nltk.

        features_name : str or list
            Which features to use. Can be:
                * 'ZhangJohnson': The features used in Zhang and Johnson (2003).
                * 'word_embedding': Word embedding only.
                * a list containing any combination of the above options.

        src_train, tgt_train, tgt_test : lists
            Each of these is a list of lists, with entries of the form:
                    ( (word, pos, domain), entity )
            For now tgt_test is needed as an argument in order to get the full
            vocabulary for word embeddings.

        **kwargs:
            if classifier_name is 'averaged_perceptron':
                'num_iterations', default: 5 (same as nltk's default)
            if classifier_name is 'megam', 'IIS', or 'GIS':
                'gauss_prior_sigma', default: 1.0 (same as nltk's default)

        References
        ----------
        [1] Daumé III, Hal. "Frustratingly easy domain adaptation." arXiv
            preprint arXiv:0907.1815 (2009).

        [2] L-BFGS: http://aria42.com/blog/2014/12/understanding-lbfgs
                    http://www.umiacs.umd.edu/~hal/docs/daume04cg-bfgs.pdf

        Remarks
        -------
        On speed and memory:
            * megam is slow and memory intensive, though using the optimized
              megam (megam opt) does help.
            * IIS and GIS are pure python and slower than megam.

        """

        #######################################################################
        ## Store model information
        #######################################################################


        self.transfer_method = transfer_method
        self.parameters = kwargs

        if not self.instance_variables:
            self.instance_variables = {'classifier': classifier_name, 'main_transfer_method': self.main_transfer_method,
                                   'params': kwargs, 'model': self.model, 'features_name': features_name}

        if self._verbose:
            print("Transfer Learning: ", transfer_method, "  Classifier: ", classifier_name)
        print('kwargs', kwargs)

    #######################################################################
    ## Determine which features to use
    #######################################################################
    #TODO make an option so can choose whether to augment the word-embeddings with the other features or not...
        if isinstance(features_name,str):
            features = self.get_featurefunc(features_name, transfer_method)

        if isinstance(features_name,list):
            featureslist = []
            for featname in features_name:
                f = self.get_featurefunc(featname, transfer_method)
                featureslist.append(f)
            print('Combining features...')
            features = combine_several_featfunctions(featureslist)

    #######################################################################
    ## Transfer Learning Options (specify training data & preprocessing)
    #######################################################################

        if transfer_method in ['src','tgt','all']:
            features_used = features
            if transfer_method=='src': train_data = self.src_train
            if transfer_method=='tgt': train_data = self.tgt_train
            if transfer_method=='all': train_data = self.src_train + self.tgt_train

        elif transfer_method == 'augment':
            train_data = self.src_train + self.tgt_train #self.all_train
            def augment_features(tokens, index, history):
                word, pos, domain = tokens[index]
                fts = features(tokens, index, history)
                for key in list(fts.keys()):
                    if domain == 'src':
                        fts[domain + '-' + key] = fts[key]
                        fts['tgt' + '-' + key] = 0
                    else:
                        fts[domain + '-' + key] = fts[key]
                        fts['src' + '-' + key] = 0
                return fts
            features_used = augment_features

        elif transfer_method == '_pred':
        # this is not to be called directly;
        # It is used by pred, to train the second classifier.

            train_data = self.tgt_train
            no_prefix = self.parameters.get('no_prefix')
            with_cca = self.parameters.get('with_cca')
            kdim = self.parameters.get('kdim')
            exclude_O = self.parameters.get('exclude_O')

            if with_cca:
                self.label2vec = cca(self.src_train + self.tgt_train,
                                no_prefix = no_prefix,
                                k = kdim,
                                exclude_O = exclude_O)

            features_used = self.get_pred_features(features,kdim,with_cca,no_prefix)

        elif transfer_method == 'pred':
            no_prefix = self.parameters.get('no_prefix')
            with_cca = self.parameters.get('with_cca')
            kdim = self.parameters.get('kdim')
            exclude_O = self.parameters.get('exclude_O')

            if kdim is None:
                kdim = 2

            if with_cca: self.path_addons += '_CCA'
            if no_prefix: self.path_addons += '_noPrefix'
            if exclude_O: self.path_addons += '_excludeO'
            self.src_classifier = None
            self.tgt_classifier = None


            # names of the two classifiers, in order
            classifier_name1 = classifier_name2 = classifier_name

            print('Training first classifier.')
            self.train('src', classifier_name1, features_name = features_name )

            # FIRST: Use classifier on both the tgt_test and tgt_train
            print('Tagging tgt test data.')
            test_input_sentences = [list(zip(*t))[0] for t in self.tgt_test]
            test_predsents = self.NER.tag_sents(test_input_sentences)

            # flatten them:
            test_augmented = [[tuple(list(f)+[list(zip(*p))[1][i]]) for i,f in enumerate(list(zip(*p))[0])] for p in test_predsents]
            tgt_test = [list(zip(x, [iob for (x,iob) in self.tgt_test[i]])) for i,x in enumerate(test_augmented)]

            # This is a list of lists of the form ((word, pos, dom, pred), iob)
            print('Tagging tgt train data.')
            train_input_sentences = [list(zip(*t))[0] for t in self.tgt_train]
            train_predsents = self.NER.tag_sents(train_input_sentences)
            train_augmented = [[tuple(list(f)+[list(zip(*p))[1][i]]) for i,f in enumerate(list(zip(*p))[0])] for p in train_predsents]
            self.tgt_train = [list(zip(x, [iob for (x,iob) in self.tgt_train[i]])) for i,x in enumerate(train_augmented)]

            # SECOND: train another classifier on the tgt_train data, with
            # the appended features from the first classifier.
            print('Training second classifier.\n')
            self.train('_pred', classifier_name2, features_name=features_name, kdim=kdim, no_prefix=no_prefix, with_cca=with_cca, exclude_O=exclude_O)

            classifier_name = 'none' # to prevent from continuing a second time.
            # self.predscore = self.evaluate(tgt_test)
            self.transfer_method = 'pred' # because the recursion will have changed it.

        else:
            pass

    #######################################################################
    ## Classifier Options: specifies which classifier to use and train
    #######################################################################
        # With 'megam, 'IIS', 'GIS' and 'naivebayes', will use
        # ClassifierBasedTagger to train the model.

        if classifier_name in ['IIS','GIS','naivebayes']:
            if classifier_name == 'naivebayes':
                print("Training the model now...")
                classifier = NaiveBayesClassifier.train
                # NOTE Naive bayes works poorly with augment (due to the
                # breaking down of the independence assumption). This is
                # described in:
                #      Sutton and McCallum, An Introduction to Conditional
                #      Random Fields, p.16.

            if classifier_name in ['IIS','GIS']:
                print("Training the model now...")

                # NOTE: Though GIS and IIS cases also take gaussian_prior_sigma,
                #       they don't use it.  It only applies to megam.
                self._set_parameter('gauss_prior_sigma', classifier_name, 1.0)
                gauss_prior_sigma = self.parameters['gauss_prior_sigma']

                classifier = lambda traindata: MaxentClassifier.train(traindata, algorithm = classifier_name, gaussian_prior_sigma = gauss_prior_sigma, trace = 3*self._verbose)

            self.model_params = {'algorithm': classifier_name, 'classifier': classifier}
            if self.label2vec: self.model_params['label2vec'] = self.label2vec
            self.NER = ClassifierBasedTagger(train=train_data,feature_detector=features_used, classifier_builder=classifier,verbose=self._verbose,)

        if classifier_name == 'averaged_perceptron':
            print("Training the model now...")
            # TODO revert
            # self._set_parameter('num_iterations', classifier_name, 5)
            self._set_parameter('num_iterations', classifier_name, 1)
            num_iter = self.parameters['num_iterations']

            self.NER = PerceptronNER(feature_detector = features_used, verbose = self._verbose)
            self.model_params = {'num_terations':num_iter}
            if self.label2vec: self.model_params['label2vec'] = self.label2vec
            self.NER.train(train_data, num_iterations = num_iter)

        if classifier_name == 'CRF':
            crfalgorithm = self.parameters.get('algorithm')
            if crfalgorithm is None:
                crfalgorithm = 'lbfgs' #'l2sgd'
                self.parameters['algorithm'] = crfalgorithm
            else:
                if crfalgorithm not in {'l2sgd', 'lbfgs'}:
                    raise ValueError("algorithm must be l2sgd' or 'lbfgs'.")

            print("Training the model now...")
            # more training options possible.
            # 'lbfgs' #'l2sgd' # lbfgs
            self.NER = CRFTagger(feature_detector = features_used, verbose = self._verbose, algorithm = crfalgorithm)
            self.NER.train(train_data, 'model.crf.tagger')

            self.model_params = {'algorithm' : crfalgorithm}
            if self.label2vec: self.model_params['label2vec'] = self.label2vec

        if self.main_transfer_method == 'pred':
            if transfer_method == 'src':
                self.src_classifier = self.NER
            elif transfer_method == '_pred':
                self.tgt_classifier = self.NER

        if classifier_name != 'none' and self.model_params and self.instance_variables:
            self.save_model(classifier_name, transfer_method, model_params=self.model_params, instance_variables=self.instance_variables)

        if classifier_name not in {'CRF', 'averaged_perceptron', 'megam',
                                   'IIS', 'GIS', 'naivebayes', 'none'}:
            raise ValueError("Wrong classifier name.")

    def predict(self, tgt_test):
        tgt_test = utils.attach_domain(tgt_test, 'tgt')

        if self.main_transfer_method == 'pred':

            print('Predicting for PRED...')
            test_input_sentences = [list(zip(*t))[0] for t in tgt_test]
            test_predsents = self.src_classifier.tag_sents(test_input_sentences)
            # flatten them:
            test_augmented = [[tuple(list(f) + [list(zip(*p))[1][i]]) for i, f in enumerate(list(zip(*p))[0])] for p
                              in test_predsents]
            tgt_test = [list(zip(x, [iob for (x, iob) in tgt_test[i]])) for i, x in enumerate(test_augmented)]

            # Step 2: load target and pred
            test_input_sentences = [list(zip(*t))[0] for t in tgt_test]
            predicted = self.tgt_classifier.tag_sents(test_input_sentences)

        else:
            test_input_sentences = [list(zip(*t))[0] for t in tgt_test]
            predicted = self.NER.tag_sents(test_input_sentences)

        return predicted

    def eval(self, predicted, actual):
        tagset_src = self.model['entity_types_src'] if 'entity_types_src' in self.model else []
        E = Evaluator(predicted, actual, tagset_src)
        self.log_results(E)
        return E

    def save_model(self, classifier_name, transfer_method, model_params, instance_variables):
        directory_name = 'models/' + self.dataset+'_' + self.main_classifier_name + '_' + self.main_transfer_method + self.path_addons

        utils.save_params(instance_variables, name=INSTANCE_VAR_NAME, directory=directory_name)
        utils.save_params(model_params, name=transfer_method + MODEL_PARAMS_NAME, directory=directory_name)
        if classifier_name == 'CRF':
            # Save differently for CRF
            utils.copy_CRF_model(name=transfer_method, directory=directory_name)
        else:
            # Save regular for IIS, GIS, megam, perceptron
            utils.save_model(self.NER, name=transfer_method + TRAINED_MODEL_NAME, directory=directory_name)

    def fetch_stored_models(self):
        return utils.fetch_model_list('models')

    def load_model(self, directory='models/current'):
        instance_variables = utils.load_params(INSTANCE_VAR_NAME, directory=directory)
        self.main_classifier_name = classifier = instance_variables['classifier']
        self.main_transfer_method = transfer_method = instance_variables['main_transfer_method']
        self.model = instance_variables['model']
        self.parameters = instance_variables['params']
        features_name = instance_variables['features_name']

        if transfer_method == 'pred':
            no_prefix = self.parameters.get('no_prefix')
            with_cca = self.parameters.get('with_cca')
            kdim = self.parameters.get('kdim')
            exclude_O = self.parameters.get('exclude_O')

            if with_cca: self.path_addons += '_CCA'
            if no_prefix: self.path_addons += '_noPrefix'
            if exclude_O: self.path_addons += '_excludeO'

            if classifier == 'CRF':
                features = self.get_featurefunc(features_name=features_name,transfer_method='src')
                src_model = os.path.join(directory, 'src.tagger')
                src_model_params = utils.load_params('src' + MODEL_PARAMS_NAME, directory=directory)
                self.src_classifier = CRFTagger(feature_detector=features, verbose=self._verbose, algorithm=src_model_params['algorithm'])
                self.src_classifier.set_model_file(src_model)

                features = self.get_featurefunc(features_name=features_name, transfer_method='src')
                features_used = self.get_pred_features(features, kdim, with_cca, no_prefix)
                tgt_model =os.path.join(directory, '_pred.tagger')
                tgt_model_params = utils.load_params('_pred' + MODEL_PARAMS_NAME, directory=directory)
                self.tgt_classifier = CRFTagger(feature_detector=features_used, verbose=self._verbose, algorithm=tgt_model_params['algorithm'])
                self.tgt_classifier.set_model_file(tgt_model)
                if with_cca: self.label2vec = tgt_model_params['label2vec']

            else:
                self.src_classifier = utils.load_params('src' + TRAINED_MODEL_NAME, directory=directory)
                self.tgt_classifier = utils.load_params('_pred' + TRAINED_MODEL_NAME, directory=directory)
        else:

            if classifier == 'CRF':
                features = self.get_featurefunc(features_name=features_name, transfer_method=transfer_method)
                src_model = os.path.join(directory, transfer_method+'.tagger')
                src_model_params = utils.load_params(transfer_method + MODEL_PARAMS_NAME, directory=directory)
                self.src_classifier = CRFTagger(feature_detector=features, verbose=self._verbose, algorithm=src_model_params['algorithm'])
                self.NER.set_model_file(src_model)
            else:
                self.NER = utils.load_params(transfer_method + TRAINED_MODEL_NAME, directory=directory)

    # Other functions
    def _set_parameter(self,paramname, classifier_name, defaultvalue):
        """ Raise ValueError if the wrong parameter name (paramname) is given and
        the dictionary self.parameters in not empty, and
        save the new parameters into self.parameters. If no parameters are given
        the defaultvalue is used.

        This is used by train method of DomainAdaptation to set default parameters.

        Parameters
        ----------
        paramname : str, name of parameter to set
        defaultvalue : the default value
        """
        # if paramname not in self.parameters and len(list(self.parameters.keys()))!=0:
        #     self.parameters={}
        #     raise ValueError('Optional argument for '+ classifier_name + ' must be '+ paramname)
        # else:
        # param = self.parameters.get(paramname, defaultvalue)
        self.parameters[paramname] = defaultvalue

    def get_featurefunc(self, features_name, transfer_method):
        if features_name == 'ZhangJohnson':
            features = ZhangJohnson
        elif features_name == 'word_embedding':
            if transfer_method == 'src': allsentences = self.src_train + self.tgt_test
            if transfer_method == 'tgt': allsentences = self.tgt_train + self.tgt_test
            if transfer_method == '_pred': allsentences = self.tgt_train + self.tgt_test
            if transfer_method in ['all', 'augment', 'pred']:
                allsentences = self.src_train + self.tgt_train + self.tgt_test
            allwords = get_unique_words(allsentences)
            print('Obtaining word embedding information.')
            wordEmbeddings, word2Idx = get_word_embeddings(embeddingsPath, allwords)
            features = wordembeddings_as_features(wordEmbeddings, word2Idx)
            print('Done obtaining word embeddings to use as features.')
        else:
            raise ValueError("features name is incorrect.")
        return features

    def log_results(self, E):

        directory = 'models/' + self.dataset+'_' + self.main_classifier_name + '_' + self.main_transfer_method + self.path_addons
        writefile(E.predicted, directory, 'predicted.conll', sep=' ')
        E.write_report(os.path.join(directory, 'results.txt'), self.excluded_entities)

    def get_pred_features(self,features, kdim, with_cca, no_prefix):

        def pred_features(tokens, index, history):
            PRED = tokens[index][3]
            fts = features(tokens, index, history)
            if with_cca:
                for i in range(kdim):
                    fts['PRED-cca-' + str(i)] = self.label2vec[PRED][i]
            else:
                fts['PRED'] = PRED

            return fts

        def pred_features_noprefix(tokens, index, history):
            PRED = tokens[index][3]
            fts = features(tokens, index, history)
            # remove prefix 'I-' or 'B-':
            if PRED != 'O':
                PRED = PRED[2:]

            if with_cca:
                for i in range(kdim):
                    fts['PRED-cca-' + str(i)] = self.label2vec[PRED][i]
            else:
                fts['PRED'] = PRED

            return fts

        if no_prefix:
            return pred_features_noprefix
        else:
            return pred_features

    def save_model_external(self, location='models/current'):
        directory_name = 'models/' + self.dataset + '_' + self.main_classifier_name + '_' + self.main_transfer_method + self.path_addons
        utils.update_recent_model(directory_name, location)

    def load_model_external(self, location='models/current'):
        self.load_model(location)
