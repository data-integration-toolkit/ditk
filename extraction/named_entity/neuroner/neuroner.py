import codecs
import copy
import glob
import os
import pickle
import random
import shutil
import time
import sklearn
import spacy
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from ner import Ner
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import train
import dataset
from entity_lstm import EntityLSTM
import utils
import conll_to_brat
import evaluate
import brat_to_conll
import utils_nlp
import prepare_pretrained_model

class neuroner(Ner):

    prediction_count = 0

    def __init__(self, **kwargs):
        """
        Set parameters for class variables
        """
        # Set parameters
        self.parameters, self.conf_parameters = utils.load_parameters(**kwargs)
        self.loadmodel = False
        self.training = False

    def _create_stats_graph_folder(self, parameters):
        """
        Initialize stats_graph_folder.

        Args:
            parameters (dict): {key: value}, dictionary with parameters as key along with their corresponding value
        Returns:
            stats_graph_folder (str) : path to the folder created
        Raises:
            None
        """
        experiment_timestamp = utils.get_current_time_in_miliseconds()
        dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder'])
        model_name = '{0}_{1}'.format(dataset_name, experiment_timestamp)
        utils.create_folder_if_not_exists(parameters['output_folder'])

        # Folder where to save graphs
        stats_graph_folder = os.path.join(parameters['output_folder'], model_name)
        utils.create_folder_if_not_exists(stats_graph_folder)
        return stats_graph_folder, experiment_timestamp

    def _get_valid_dataset_filepaths(self, parameters, dataset_types=['train', 'valid', 'test']):
        """
        Get paths for the datasets. Also converts the dataset from CoNLL format to BRAT (specific to implmentation) format if not already done.

        Args:
            parameters (dict): {key: value}, dictionary with parameters as key along with their corresponding value
            dataset_types (list): types of datasets given eg. train dataset, test dataset etc.
        Returns:
            dataset_filepaths (dict) : {key: value}, dictionary with key as type (train, valid, test) of dataset and value as file paths with corresponding data in CoNLL format
            dataset_brat_folders (dict) : {key: value}, dictionary with key as type (train, valid, test) of dataset and value as folders with corresponding data in BRAT format
        Raises:
            None
        """
        dataset_filepaths = {}
        dataset_brat_folders = {}

        for dataset_type in dataset_types:
            dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'],
                '{0}.txt'.format(dataset_type))
            if not os.path.isfile(dataset_filepaths[dataset_type]):
                dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'],
                                                               self.filename)
            dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'],
                dataset_type)
            dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'],
                '{0}_compatible_with_brat.txt'.format(dataset_type))

            # Conll file exists
            if os.path.isfile(dataset_filepaths[dataset_type]) \
            and os.path.getsize(dataset_filepaths[dataset_type]) > 0:
                # Brat text files exist
                if os.path.exists(dataset_brat_folders[dataset_type]) and \
                len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                    # Check compatibility between conll and brat files
                    brat_to_conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                    if os.path.exists(dataset_compatible_with_brat_filepath):
                        dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath
                    conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type],
                        dataset_brat_folders[dataset_type])

                # Brat text files do not exist
                else:
                    # Populate brat text and annotation files based on conll file
                    conll_to_brat.conll_to_brat(dataset_filepaths[dataset_type],
                        dataset_compatible_with_brat_filepath, dataset_brat_folders[dataset_type],
                        dataset_brat_folders[dataset_type])
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

            # Conll file does not exist
            else:
                # Brat text files exist
                if os.path.exists(dataset_brat_folders[dataset_type]) \
                and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
                    dataset_filepath_for_tokenizer = os.path.join(parameters['dataset_text_folder'],
                        '{0}_{1}.txt'.format(dataset_type, parameters['tokenizer']))
                    if os.path.exists(dataset_filepath_for_tokenizer):
                        conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepath_for_tokenizer,
                            dataset_brat_folders[dataset_type])
                    else:
                        # Populate conll file based on brat files
                        brat_to_conll.brat_to_conll(dataset_brat_folders[dataset_type],
                            dataset_filepath_for_tokenizer, parameters['tokenizer'], parameters['spacylanguage'])
                    dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

                # Brat text files do not exist
                else:
                    del dataset_filepaths[dataset_type]
                    del dataset_brat_folders[dataset_type]
                    continue

            if parameters['tagging_format'] == 'bioes':
                # Generate conll file with BIOES format
                bioes_filepath = os.path.join(parameters['dataset_text_folder'],
                    '{0}_bioes.txt'.format(utils.get_basename_without_extension(dataset_filepaths[dataset_type])))
                utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type],
                    bioes_filepath)
                dataset_filepaths[dataset_type] = bioes_filepath

        return dataset_filepaths, dataset_brat_folders

    def _check_param_compatibility(self, parameters, dataset_filepaths):
        """
        Check whether parameters are compatible.

        Args:
            parameters (dict): {key: value}, dictionary with parameters as key along with their corresponding value
            dataset_filepaths (type): description.
        Returns:
            None
        Raises:
            None
        """
        utils.check_param_compatibility(parameters, dataset_filepaths)

    def predict_text(self, text):
        """
        Makes predictions on a given input text.
        Args:
            text (str): string of text on which to make prediction
        Returns:
            predictions: [tuple,...], i.e. list of tuples.
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
                 NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                     evalutation. See note in evaluate() about parallel arrays.]
        Raises:
            None
        """
        # IMPLEMENT PREDICTION.
        self.prediction_count += 1

        if self.prediction_count == 1:
            self.parameters['dataset_text_folder'] = os.path.join('.', 'data', 'temp')
            self.stats_graph_folder, _ = self._create_stats_graph_folder(self.parameters)

        # Update the deploy folder, file, and modeldata
        dataset_type = 'deploy'

        # Delete all deployment data
        for filepath in glob.glob(os.path.join(self.parameters['dataset_text_folder'],
                                               '{0}*'.format(dataset_type))):
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)

        # Create brat folder and file
        dataset_brat_deploy_folder = os.path.join(self.parameters['dataset_text_folder'],
                                                  dataset_type)
        utils.create_folder_if_not_exists(dataset_brat_deploy_folder)
        dataset_brat_deploy_filepath = os.path.join(dataset_brat_deploy_folder,
                                                    'temp_{0}.txt'.format(str(self.prediction_count).zfill(5)))
        # self._get_dataset_brat_deploy_filepath(dataset_brat_deploy_folder)
        with codecs.open(dataset_brat_deploy_filepath, 'w', 'UTF-8') as f:
            f.write(text)

        # Update deploy filepaths
        dataset_filepaths, dataset_brat_folders = self._get_valid_dataset_filepaths(self.parameters,
                                                                                    dataset_types=[dataset_type])
        self.dataset_filepaths.update(dataset_filepaths)
        self.dataset_brat_folders.update(dataset_brat_folders)

        # Update the dataset for the new deploy set
        self.modeldata.update_dataset(self.dataset_filepaths, [dataset_type])

        # Predict labels and output brat
        output_filepaths = {}
        prediction_output = train.prediction_step(self.sess, self.modeldata,
                                                  dataset_type, self.model, self.transition_params_trained,
                                                  self.stats_graph_folder, self.prediction_count, self.parameters,
                                                  self.dataset_filepaths)

        _, _, output_filepaths[dataset_type] = prediction_output
        conll_to_brat.output_brat(output_filepaths, self.dataset_brat_folders,
                                  self.stats_graph_folder, overwrite=True)

        # Print and output result
        text_filepath = os.path.join(self.stats_graph_folder, 'brat', 'deploy',
                                     os.path.basename(dataset_brat_deploy_filepath))
        annotation_filepath = os.path.join(self.stats_graph_folder, 'brat',
                                           'deploy', '{0}.ann'.format(
                utils.get_basename_without_extension(dataset_brat_deploy_filepath)))
        text2, entities = brat_to_conll.get_entities_from_brat(text_filepath,
                                                               annotation_filepath, verbose=True)

        if self.parameters['tokenizer'] == 'spacy':
            spacy_nlp = spacy.load(self.parameters['spacylanguage'])

        tokens = spacy_nlp(text)
        predictions = []
        pred_tuple = prediction_output[0]

        for i, token in enumerate(tokens):
            pred = (token.idx, len(token), token, self.modeldata.index_to_label[pred_tuple[i]])
            predictions.append(pred)

        assert (text == text2)
        return predictions

    def predict_dataset(self, data, dataset_type='test'):
        """
        Makes predictions on a given dataset and returns the predictions in the specified format
        Args:
            dataset: data in arbitrary format as required for testing
        Returns:
            predictions: [tuple,...], i.e. list of tuples.
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
                 NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                     evalutation. See note in evaluate() about parallel arrays.]
        Raises:
            None
        """
        # IMPLEMENT PREDICTION.

        # Load dataset only when directly loading the model

        if self.training:
            self.parameters['use_pretrained_model'] = True
            self.parameters['pretrained_model_folder'] = os.path.join('.', 'output', os.path.basename(self.stats_graph_folder), 'output')
            tf.reset_default_graph()

        if self.loadmodel:
            self.dataset_filepaths, self.dataset_brat_folders = self._get_valid_dataset_filepaths(self.parameters)
            self.modeldata = dataset.Dataset(verbose=self.parameters['verbose'], debug=self.parameters['debug'])
            self.token_to_vector = self.modeldata.load_dataset(data, self.dataset_filepaths, self.parameters)
            self.loadmodel = False

        # Launch session. Automatically choose a device
        # if the specified one doesn't exist
        if self.parameters['use_pretrained_model']:
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=self.parameters['number_of_cpu_threads'],
                inter_op_parallelism_threads=self.parameters['number_of_cpu_threads'],
                device_count={'CPU': 1, 'GPU': self.parameters['number_of_gpus']},
                allow_soft_placement=True,
                log_device_placement=False)

            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():

                # Initialize or load pretrained model
                self.model = EntityLSTM(self.modeldata, self.parameters)
                self.sess.run(tf.global_variables_initializer())

                self.transition_params_trained = self.model.restore_from_pretrained_model(self.parameters,
                                                                                              self.modeldata, self.sess,
                                                                                              token_to_vector=self.token_to_vector)

        parameters = self.parameters
        dataset_filepaths = self.dataset_filepaths
        sess = self.sess
        model = self.model
        transition_params_trained = self.transition_params_trained
        stats_graph_folder, experiment_timestamp = self._create_stats_graph_folder(self.parameters)

        all_predictions = []
        true_labels = []
        tokens = []
        span = []

        output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type,
                                                                                    0))
        output_file = codecs.open(output_filepath, 'w', 'UTF-8')
        original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'UTF-8')

        modeldata = self.modeldata

        for i in range(len(modeldata.token_indices[dataset_type])):
            feed_dict = {
                model.input_token_indices: modeldata.token_indices[dataset_type][i],
                model.input_token_character_indices: modeldata.character_indices_padded[dataset_type][i],
                model.input_token_lengths: modeldata.token_lengths[dataset_type][i],
                model.input_label_indices_vector: modeldata.label_vector_indices[dataset_type][i],
                model.dropout_keep_prob: 1.
            }

            unary_scores, predictions = sess.run([model.unary_scores,
                                                  model.predictions], feed_dict)

            if parameters['use_crf']:
                predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores,
                                                               transition_params_trained)
                predictions = predictions[1:-1]
            else:
                predictions = predictions.tolist()

            assert (len(predictions) == len(modeldata.tokens[dataset_type][i]))

            output_string = ''
            prediction_labels = [modeldata.index_to_label[prediction] for prediction in predictions]
            unary_score_list = unary_scores.tolist()[1:-1]

            gold_labels = modeldata.labels[dataset_type][i]

            if parameters['tagging_format'] == 'bioes':
                prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
                gold_labels = utils_nlp.bioes_to_bio(gold_labels)

            for prediction, token, gold_label, scores in zip(prediction_labels,
                                                             modeldata.tokens[dataset_type][i], gold_labels,
                                                             unary_score_list):

                while True:
                    line = original_conll_file.readline()
                    split_line = line.strip().split(' ')

                    if '-DOCSTART-' in split_line[0] or len(split_line) == 0 \
                            or len(split_line[0]) == 0:
                        continue
                    else:
                        token_original = split_line[0]

                        if parameters['tagging_format'] == 'bioes':
                            split_line.pop()

                        gold_label_original = split_line[6]

                        assert (token == token_original and gold_label == gold_label_original)
                        break

                split_line.append(prediction)
                if parameters['output_scores']:
                    # space separated scores
                    scores = ' '.join([str(i) for i in scores])
                    split_line.append('{}'.format(scores))
                output_string += ' '.join(split_line) + '\n'

            output_file.write(output_string + '\n')

            predicted_labels = [modeldata.index_to_label[preds] for preds in predictions]

            all_predictions.extend(prediction_labels)
            all_predictions.append('')
            true_labels.extend(modeldata.labels[dataset_type][i])
            true_labels.append('')
            tokens.extend(modeldata.tokens[dataset_type][i])
            tokens.append('')
            span.extend(modeldata.token_lengths[dataset_type][i])
            span.append('')

        start_index = [None] * len(true_labels)

        all_predictions = list(map(list, zip(start_index, span, tokens, all_predictions)))
        all_predictions = [tuple(pred) for pred in all_predictions]

        return all_predictions

    #@overrides(DITKModel_NER)
    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        Converts test data into common format for evaluation [i.e. same format as predict()]
        This added step/layer of abstraction is required due to the refactoring of read_dataset_traint()
        and read_dataset_test() back to the single method of read_dataset() along with the requirement on
        the format of the output of predict() and therefore the input format requirement of evaluate(). Since
        individuals will implement their own format of data from read_dataset(), this is the layer that
        will convert to proper format for evaluate().
        Args:
            data: data in proper [arbitrary] format for train or test. [i.e. format of output from read_dataset]
        Returns:
            ground_truth: [tuple,...], i.e. list of tuples. [SAME format as output of predict()]
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
        Raises:
            None
        """
        # IMPLEMENT CONVERSION. STRICT OUTPUT FORMAT REQUIRED.


        # Load dataset
        if self.loadmodel:
            self.dataset_filepaths, self.dataset_brat_folders = self._get_valid_dataset_filepaths(self.parameters)
            self.modeldata = dataset.Dataset(verbose=self.parameters['verbose'], debug=self.parameters['debug'])
            self.token_to_vector = self.modeldata.load_dataset(data, self.dataset_filepaths, self.parameters)
            self.loadmodel = False

        modeldata = self.modeldata

        # return ground_truth
        true_labels = []
        tokens = []
        span = []

        dataset_type = 'test'

        if len(args)==1:
            dataset_type = args[0]

        for i in range(len(modeldata.token_indices[dataset_type])):
            true_labels.extend(modeldata.labels[dataset_type][i])
            true_labels.append('')
            tokens.extend(modeldata.tokens[dataset_type][i])
            tokens.append('')
            span.extend(modeldata.token_lengths[dataset_type][i])
            span.append('')

        start_index = [None] * len(true_labels)

        all_y_true = list(map(list, zip(start_index, span, tokens, true_labels)))
        all_y_true = [tuple(pred) for pred in all_y_true]

        return all_y_true

    def save_model(self, file=None):
        """
        :param file: Where to save the model - Optional function
        :return:
        """
        utils.create_folder_if_not_exists(self.modelFolder)
        self.model.saver.save(self.sess, os.path.join(self.modelFolder, 'model_{0:05d}.ckpt'.format(0)))

    def load_model(self, file=None):
        """
        :param file: From where to load the model - Optional function
        :return:
        """

        self.parameters['use_pretrained_model'] = True
        self.parameters['pretrained_model_folder'] = file if file!=None else self.parameters['pretrained_model_folder']
        self.loadmodel = True


    #@overrides(DITKModel_NER)
    def read_dataset(self, file_dict, dataset_name=None, *args, **kwargs):  # <--- implemented PER class
        """
        Reads a dataset in preparation for train or test. Returns data in proper format for train or test.
        Args:
            fileNames: list-like. List of files representing the dataset to read. Each element is str, representing
                filename [possibly with filepath]
        Returns:
            data: data in arbitrary format for train or test.
        Raises:
            None
        """
        # IMPLEMENT READING
        # pass
        standard_split = ["train", "test", "dev"]
        dataset_root = os.path.dirname(file_dict['train'])
        self.parameters['dataset_text_folder'] = dataset_root
        self.filename = os.path.basename(file_dict['train'])
        data = {}

        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()
                for i, line in enumerate(raw_data):
                    if len(line.strip()) > 0:
                        raw_data[i] = line.strip().split()
                    else:
                        raw_data[i] = list(line)
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        return data

    #@overrides(DITKModel_NER)
    def train(self, data, *args, **kwargs):
        """
        Trains a model on the given input data
        Args:
            data: iterable of arbitrary format. represents the data instances and features and labels you use to train your model.
        Returns:
            ret: None. Trained model stored internally to class instance state.
        Raises:
            None
        """
        # IMPLEMENT TRAINING.
        # pass

        self.dataset_filepaths, self.dataset_brat_folders = self._get_valid_dataset_filepaths(self.parameters)
        self._check_param_compatibility(self.parameters, self.dataset_filepaths)

        # Load dataset
        self.modeldata = dataset.Dataset(verbose=self.parameters['verbose'], debug=self.parameters['debug'])
        self.token_to_vector = self.modeldata.load_dataset(data, self.dataset_filepaths, self.parameters)

        # Launch session. Automatically choose a device
        # if the specified one doesn't exist
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=self.parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=self.parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': self.parameters['number_of_gpus']},
            allow_soft_placement=True,
            log_device_placement=False)

        self.sess = tf.Session(config=session_conf)
        with self.sess.as_default():

            # Initialize or load pretrained model
            self.model = EntityLSTM(self.modeldata, self.parameters)
            self.sess.run(tf.global_variables_initializer())

            if self.parameters['use_pretrained_model']:
                self.transition_params_trained = self.model.restore_from_pretrained_model(self.parameters,
                                                                                          self.modeldata, self.sess,
                                                                                          token_to_vector=self.token_to_vector)
            else:
                self.model.load_pretrained_token_embeddings(self.sess, self.modeldata,
                                                            self.parameters, self.token_to_vector)
                self.transition_params_trained = np.random.rand(len(self.modeldata.unique_labels) + 2,
                                                                len(self.modeldata.unique_labels) + 2)

        parameters = self.parameters
        conf_parameters = self.conf_parameters
        dataset_filepaths = self.dataset_filepaths
        modeldata = self.modeldata
        dataset_brat_folders = self.dataset_brat_folders
        sess = self.sess
        model = self.model
        transition_params_trained = self.transition_params_trained
        stats_graph_folder, experiment_timestamp = self._create_stats_graph_folder(parameters)

        self.stats_graph_folder = stats_graph_folder

        # Initialize and save execution details
        start_time = time.time()
        results = {}
        results['epoch'] = {}
        results['execution_details'] = {}
        results['execution_details']['train_start'] = start_time
        results['execution_details']['time_stamp'] = experiment_timestamp
        results['execution_details']['early_stop'] = False
        results['execution_details']['keyboard_interrupt'] = False
        results['execution_details']['num_epochs'] = 0
        results['model_options'] = copy.copy(parameters)

        model_folder = os.path.join(stats_graph_folder, 'model')
        self.modelFolder = model_folder
        utils.create_folder_if_not_exists(model_folder)
        with open(os.path.join(model_folder, 'parameters.ini'), 'w') as parameters_file:
            conf_parameters.write(parameters_file)
        pickle.dump(modeldata, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))

        tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
        utils.create_folder_if_not_exists(tensorboard_log_folder)
        tensorboard_log_folders = {}
        for dataset_type in dataset_filepaths.keys():
            tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder,
                                                                 'tensorboard_logs', dataset_type)
            utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])

        # Instantiate the writers for TensorBoard
        writers = {}
        for dataset_type in dataset_filepaths.keys():
            writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type],
                                                          graph=sess.graph)

        # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings
        embedding_writer = tf.summary.FileWriter(model_folder)

        embeddings_projector_config = projector.ProjectorConfig()
        tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
        tensorboard_token_embeddings.tensor_name = model.token_embedding_weights.name
        token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
        tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '.')

        tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
        tensorboard_character_embeddings.tensor_name = model.character_embedding_weights.name
        character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv')
        tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '.')

        projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

        # Write metadata for TensorBoard embeddings
        token_list_file = codecs.open(token_list_file_path, 'w', 'UTF-8')
        for token_index in range(modeldata.vocabulary_size):
            token_list_file.write('{0}\n'.format(modeldata.index_to_token[token_index]))
        token_list_file.close()

        character_list_file = codecs.open(character_list_file_path, 'w', 'UTF-8')
        for character_index in range(modeldata.alphabet_size):
            if character_index == modeldata.PADDING_CHARACTER_INDEX:
                character_list_file.write('PADDING\n')
            else:
                character_list_file.write('{0}\n'.format(modeldata.index_to_character[character_index]))
        character_list_file.close()

        # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
        # number of epochs with no improvement on the validation test in terms of F1-score
        bad_counter = 0
        previous_best_valid_f1_score = 0
        epoch_number = -1
        try:
            while True:
                step = 0
                epoch_number += 1
                print('\nStarting epoch {0}'.format(epoch_number))

                epoch_start_time = time.time()

                if epoch_number != 0:
                    # Train model: loop over all sequences of training set with shuffling
                    sequence_numbers = list(range(len(modeldata.token_indices['train'])))
                    random.shuffle(sequence_numbers)
                    for sequence_number in sequence_numbers:
                        transition_params_trained = train.train_step(sess, modeldata,
                                                                     sequence_number, model, parameters)
                        step += 1
                        if step % 10 == 0:
                            print('Training {0:.2f}% done'.format(step / len(sequence_numbers) * 100),
                                  end='\r', flush=True)

                epoch_elapsed_training_time = time.time() - epoch_start_time
                print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time),
                      flush=True)

                y_pred, y_true, output_filepaths = train.predict_labels(sess, model,
                                                                        transition_params_trained, parameters,
                                                                        modeldata, epoch_number,
                                                                        stats_graph_folder, dataset_filepaths)

                # Evaluate model: save and plot results
                #evaluate.evaluate_model(results, modeldata, y_pred, y_true, stats_graph_folder,
                #                        epoch_number, epoch_start_time, output_filepaths, parameters)

                # Save model
                model.saver.save(sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(epoch_number)))

                # Save TensorBoard logs
                summary = sess.run(model.summary_op, feed_dict=None)
                writers['train'].add_summary(summary, epoch_number)
                writers['train'].flush()
                utils.copytree(writers['train'].get_logdir(), model_folder)

                # Early stop
                '''
                valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['micro']
                if valid_f1_score > previous_best_valid_f1_score:
                    bad_counter = 0
                    previous_best_valid_f1_score = valid_f1_score
                    conll_to_brat.output_brat(output_filepaths, dataset_brat_folders,
                                              stats_graph_folder, overwrite=True)
                    self.transition_params_trained = transition_params_trained
                else:
                    bad_counter += 1
                print("The last {0} epochs have not shown improvements on the validation set.".format(bad_counter))
                '''

                if bad_counter >= parameters['patience']:
                    print('Early Stop!')
                    results['execution_details']['early_stop'] = True
                    break

                if epoch_number >= parameters['maximum_number_of_epochs']:
                    break

        except KeyboardInterrupt:
            results['execution_details']['keyboard_interrupt'] = True
            print('Training interrupted')

        print('Finishing the experiment')
        end_time = time.time()
        results['execution_details']['train_duration'] = end_time - start_time
        results['execution_details']['train_end'] = end_time
        evaluate.save_results(results, stats_graph_folder)
        self.training = True
        main_folder = os.path.basename(stats_graph_folder)
        prepare_pretrained_model.prepare_pretrained_model_for_restoring(main_folder, epoch_number, 'output', False)
        for dataset_type in dataset_filepaths.keys():
            writers[dataset_type].close()

    #@overrides(DITKModel_NER)
    def predict(self, data, *args, **kwargs):
        """
        Predicts on the given input data. Assumes model has been trained with train()
        Args:
            data: iterable of arbitrary format. represents the data instances and features you use to make predictions
                Note that prediction requires trained model. Precondition that class instance already stores trained model
                information.
        Returns:
            predictions: [tuple,...], i.e. list of tuples.
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
                 NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                     evalutation. See note in evaluate() about parallel arrays.]
        Raises:
            None
        """
        # IMPLEMENT PREDICTION. STRICT OUTPUT FORMAT REQUIRED.

        # return predictions

        if isinstance(data, str):
            return self.predict_text(data)
        else:
            if len(args)==0:
                return self.predict_dataset(data)
            else:
                return self.predict_dataset(data, args[0])

    #@overrides(DITKModel_NER)
    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        """
        Calculates evaluation metrics on chosen benchmark dataset [Precision,Recall,F1, or others...]
        Args:
            predictions: [tuple,...], list of tuples [same format as output from predict]
            groundTruths: [tuple,...], list of tuples representing ground truth.
        Returns:
            metrics: tuple with (p,r,f1). Each element is float.
        Raises:
            None
        """
        # pseudo-implementation
        # we have a set of predictions and a set of ground truth data.
        # calculate true positive, false positive, and false negative
        # calculate Precision = tp/(tp+fp)
        # calculate Recall = tp/(tp+fn)
        # calculate F1 using precision and recall

        # return (precision, recall, f1)
        predicted_labels = [predicted[3] for predicted in predictions if predicted[3]!='']
        ground_labels = [true_labels[3] for true_labels in groundTruths if true_labels[3]!='']

        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_set = list(self.modeldata.label_to_index.keys())
        label_encoder.fit(label_set)

        ground_labels = label_encoder.transform(ground_labels)
        predicted_labels = label_encoder.transform(predicted_labels)

        new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = evaluate.remap_labels(predicted_labels,
                                                                                                 ground_labels,
                                                                                                 self.modeldata,
                                                                                                 self.parameters[
                                                                                                     'main_evaluation_mode'])

        print(sklearn.metrics.classification_report(new_y_true, new_y_pred,
                                                    digits=4, labels=new_label_indices, target_names=new_label_names))
        precision = sklearn.metrics.precision_score(new_y_true, new_y_pred, average='micro', labels=new_label_indices)
        recall = sklearn.metrics.recall_score(new_y_true, new_y_pred, average='micro', labels=new_label_indices)
        f1 = sklearn.metrics.f1_score(new_y_true, new_y_pred, average='micro', labels=new_label_indices)
        return precision, recall, f1

    def get_params(self):
        """
        Returns set parameters.

        Args:
            None.
        Returns:
            parameters (dict) : {key: value}, dictionary with parameters as key along with their corresponding value
        Raises:
            None
        """
        return self.parameters

    def close(self):
        """
        Clean up helper function

        Args:
            None.
        Returns:
            None.
        Raises:
            None
        """
        self.__del__()

    def __del__(self):
        """
        Deletes tensorflow session.

        Args:
            None.
        Returns:
            None.
        Raises:
            None
        """
        self.sess.close()


def main(train_file, dev_file=None, test_file=None):

    if dev_file==None and test_file==None:
        inputFiles = {'train': train_file,
                      'dev': train_file,
                      'test': train_file}
    else:
        inputFiles = {'train': train_file,
                      'dev': dev_file if dev_file!=None else train_file,
                      'test': test_file if test_file!=None else train_file}

    # instatiate the class
    ner = neuroner(parameters_filepath='./parameters.ini')

    # read in a dataset for training
    data = ner.read_dataset(inputFiles)

    # trains the model and stores model state in object properties or similar
    ner.train(data)

    # get ground truth from data for test set
    ground = ner.convert_ground_truth(data)

    # generate predictions on test
    predictions = ner.predict(data)

    # calculate Precision, Recall, F1
    P,R,F1 = ner.evaluate(predictions, ground)

    print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))

    output_file = os.path.dirname(train_file)
    output_file_path = os.path.join(output_file, "output.txt")

    with open(output_file_path, 'w') as f:
        for index, (g, p) in enumerate(zip(ground, predictions)):
            if index==len(predictions)-1:
                continue
            if len(g[3])==0:
                f.write("\n")
            else:
                f.write(g[2] + " " + g[3] + " " + p[3] + "\n")

    return output_file_path



if __name__ == "__main__":
    train_file = "/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/train.txt"
    dev_file = "/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/valid.txt"
    test_file = "/Users/lakshya/Desktop/CSCI-548/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/conll/test.txt"
    main(train_file, dev_file, test_file)