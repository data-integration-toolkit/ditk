'''
Miscellaneous utility functions
'''
import collections
import datetime
import operator
import configparser
import distutils.util
import glob
import os
import pickle
from pprint import pprint
import random
import shutil
import time
import warnings
import pkg_resources
import conll_to_brat
import brat_to_conll
import utils_nlp

# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "dataset":
            renamed_module = "neuroner.dataset"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def fetch_model(name):
    """
    Fetch a pre-trained model and copy to a local "trained_models" folder
     If name is provided, fetch from the package folder.

    Args:
        name (str): Name of a model folder.
    """
    # get content from package and write to local dir
    # model comprises of:
    # dataset.pickle
    # model.ckpt.data-00000-of-00001
    # model.ckpt.index
    # model.ckpt.meta
    # parameters.ini
    _fetch(name, content_type="trained_models")


def fetch_data(name):
    """
    Fetch a dataset. If name is provided, fetch from the package folder. If url
    is provided, fetch from a remote location.

    Args:
        name (str): Name of a dataset.
        url (str): URL of a model folder.
    """
    # get content from package and write to local dir
    _fetch(name, content_type="data")


def _fetch(name, content_type=None):
    """
    Load data or models from the package folder.

    Args:
        name (str): name of the resource
        content_type (str): either "data" or "trained_models"

    Returns:
        fileset (dict): dictionary containing the file content
    """
    package_name = 'neuroner'
    resource_path = '/'.join((content_type, name))

    # get dirs
    root_dir = os.path.dirname(pkg_resources.resource_filename(package_name,
        '__init__.py'))
    src_dir = os.path.join(root_dir, resource_path)
    dest_dir = os.path.join('.', content_type, name)

    if pkg_resources.resource_isdir(package_name, resource_path):

        # copy from package to dest dir
        if os.path.isdir(dest_dir):
            msg = "Directory '{}' already exists.".format(dest_dir)
            print(msg)
        else:
            shutil.copytree(src_dir, dest_dir)
            msg = "Directory created: '{}'.".format(dest_dir)
            print(msg)
    else:
        msg = "{} not found in {} package.".format(name,package_name)
        print(msg)


def _get_default_param():
    """
    Get the default parameters.

    """
    param = {'pretrained_model_folder':'./trained_models/conll_2003_en',
             'dataset_text_folder':'./data/conll2003/en',
             'character_embedding_dimension':25,
             'character_lstm_hidden_state_dimension':25,
             'check_for_digits_replaced_with_zeros':True,
             'check_for_lowercase':True,
             'debug':False,
             'dropout_rate':0.5,
             'experiment_name':'experiment',
             'freeze_token_embeddings':False,
             'gradient_clipping_value':5.0,
             'learning_rate':0.005,
             'load_only_pretrained_token_embeddings':False,
             'load_all_pretrained_token_embeddings':False,
             'main_evaluation_mode':'conll',
             'maximum_number_of_epochs':100,
             'number_of_cpu_threads':8,
             'number_of_gpus':0,
             'optimizer':'sgd',
             'output_folder':'./output',
             'output_scores': False,
             'patience':10,
             'parameters_filepath': os.path.join('.','parameters.ini'),
             'plot_format':'pdf',
             'reload_character_embeddings':True,
             'reload_character_lstm':True,
             'reload_crf':True,
             'reload_feedforward':True,
             'reload_token_embeddings':True,
             'reload_token_lstm':True,
             'remap_unknown_tokens_to_unk':True,
             'spacylanguage':'en',
             'tagging_format':'bioes',
             'token_embedding_dimension':100,
             'token_lstm_hidden_state_dimension':100,
             'token_pretrained_embedding_filepath':'./data/word_vectors/glove.6B.100d.txt',
             'tokenizer':'spacy',
             'train_model':True,
             'use_character_lstm':True,
             'use_crf':True,
             'use_pretrained_model':False,
             'verbose':False}

    return param


def _get_config_param(param_filepath=None):
    """
    Get the parameters from the config file.
    """
    param = {}

    # If a parameter file is specified, load it
    if param_filepath:
        param_file_txt = configparser.ConfigParser()
        param_file_txt.read(param_filepath, encoding="UTF-8")
        nested_parameters = convert_configparser_to_dictionary(param_file_txt)

        for k, v in nested_parameters.items():
            param.update(v)

    return param, param_file_txt


def _clean_param_dtypes(param):
    """
    Ensure data types are correct in the parameter dictionary.

    Args:
        param (dict): dictionary of parameter settings.
    """

    # Set the data type
    for k, v in param.items():
        v = str(v)
        # If the value is a list delimited with a comma, choose one element at random.
        # NOTE: review this behaviour.
        if ',' in v:
            v = random.choice(v.split(','))
            param[k] = v

        # Ensure that each parameter is cast to the correct type
        if k in ['character_embedding_dimension',
            'character_lstm_hidden_state_dimension', 'token_embedding_dimension',
            'token_lstm_hidden_state_dimension', 'patience',
            'maximum_number_of_epochs', 'maximum_training_time',
            'number_of_cpu_threads', 'number_of_gpus']:
            param[k] = int(v)
        elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value']:
            param[k] = float(v)
        elif k in ['remap_unknown_tokens_to_unk', 'use_character_lstm',
            'use_crf', 'train_model', 'use_pretrained_model', 'debug', 'verbose',
            'reload_character_embeddings', 'reload_character_lstm',
            'reload_token_embeddings', 'reload_token_lstm',
            'reload_feedforward', 'reload_crf', 'check_for_lowercase',
            'check_for_digits_replaced_with_zeros', 'output_scores',
            'freeze_token_embeddings', 'load_only_pretrained_token_embeddings']:
            param[k] = distutils.util.strtobool(v)

    return param


def load_parameters(**kwargs):
    '''
    Load parameters from the ini file if specified, take into account any
    command line argument, and ensure that each parameter is cast to the
    correct type.

    Command line arguments take precedence over parameters specified in the
    parameter file.
    '''
    param = {}
    param_default = _get_default_param()

    # use parameter path if provided, otherwise use default
    try:
        if kwargs['parameters_filepath']:
            parameters_filepath = kwargs['parameters_filepath']
    except:
        parameters_filepath = param_default['parameters_filepath']

    param_config, param_file_txt = _get_config_param(parameters_filepath)

    # Parameter file settings should overwrite default settings
    for k, v in param_config.items():
        param[k] = v

    # Command line args should overwrite settings in the parameter file
    for k, v in kwargs.items():
        param[k] = v

    # Any missing args can be set to default
    for k, v in param_default.items():
        if k not in param:
            param[k] = param_default[k]

    # clean the data types
    param = _clean_param_dtypes(param)
    print(param)

    # if loading a pretrained model, set to pretrain hyperparameters
    if param['use_pretrained_model']:

        pretrain_path = os.path.join(param['pretrained_model_folder'],
            'parameters.ini')

        if os.path.isfile(pretrain_path):
            pretrain_param, _ = _get_config_param(pretrain_path)
            pretrain_param = _clean_param_dtypes(pretrain_param)

            pretrain_list = ['use_character_lstm', 'character_embedding_dimension',
                'character_lstm_hidden_state_dimension', 'token_embedding_dimension',
                'token_lstm_hidden_state_dimension', 'use_crf']

            for name in pretrain_list:
                if param[name] != pretrain_param[name]:
                    msg = """WARNING: parameter '{0}' was overwritten from '{1}' to '{2}'
                        for consistency with the pretrained model""".format(name,
                            param[name], pretrain_param[name])
                    print(msg)
                    param[name] = pretrain_param[name]
        else:
            msg = """Warning: pretraining parameter file not found."""
            print(msg)

    # update param_file_txt to reflect the overriding
    param_to_section = get_parameter_to_section_of_configparser(param_file_txt)
    for k, v in param.items():
        try:
            param_file_txt.set(param_to_section[k], k, str(v))
        except:
            pass

    if param['verbose']:
        pprint(param)

    return param, param_file_txt


def get_valid_dataset_filepaths(parameters):
    """
    Get valid filepaths for the datasets.
    """
    dataset_filepaths = {}
    dataset_brat_folders = {}

    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'],
            '{0}.txt'.format(dataset_type))
        dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'],
            dataset_type)
        dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'],
            '{0}_compatible_with_brat.txt'.format(dataset_type))

        # Conll file exists
        if os.path.isfile(dataset_filepaths[dataset_type]) \
        and os.path.getsize(dataset_filepaths[dataset_type]) > 0:

            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) \
            and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                # Check compatibility between conll and brat files
                brat_to_conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                if os.path.exists(dataset_compatible_with_brat_filepath):
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

                conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type],
                    dataset_brat_folders[dataset_type])

            # Brat text files do not exist
            else:

                # Populate brat text and annotation files based on conll file
                conll_to_brat.conll_to_brat(dataset_filepaths[dataset_type], dataset_compatible_with_brat_filepath,
                    dataset_brat_folders[dataset_type], dataset_brat_folders[dataset_type])
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
                        dataset_filepath_for_tokenizer, parameters['tokenizer'],
                        parameters['spacylanguage'])
                dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

            # Brat text files do not exist
            else:
                del dataset_filepaths[dataset_type]
                del dataset_brat_folders[dataset_type]
                continue

        if parameters['tagging_format'] == 'bioes':
            # Generate conll file with BIOES format
            bioes_filepath = os.path.join(parameters['dataset_text_folder'],
                '{0}_bioes.txt'.format(get_basename_without_extension(dataset_filepaths[dataset_type])))
            utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type],
                bioes_filepath)
            dataset_filepaths[dataset_type] = bioes_filepath

    return dataset_filepaths, dataset_brat_folders


def check_param_compatibility(parameters, dataset_filepaths):
    """
    Check parameters are compatible.
    """
    # Check mode of operation
    if parameters['train_model']:
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            msg = """If train_model is set to True, both train and valid set must exist 
                in the specified dataset folder: {0}""".format(parameters['dataset_text_folder'])
            raise IOError(msg)
    elif parameters['use_pretrained_model']:
        if 'train' in dataset_filepaths and 'valid' in dataset_filepaths:
            msg = """WARNING: train and valid set exist in the specified dataset folder, 
                but train_model is set to FALSE: {0}""".format(parameters['dataset_text_folder'])
            print(msg)
        if 'test' not in dataset_filepaths and 'deploy' not in dataset_filepaths:
            msg = """For prediction mode, either test set and deploy set must exist 
                in the specified dataset folder: {0}""".format(parameters['dataset_text_folder'])
            raise IOError(msg)
    # if not parameters['train_model'] and not parameters['use_pretrained_model']:
    else:
        raise ValueError("At least one of train_model and use_pretrained_model must be set to True.")

    if parameters['use_pretrained_model']:
        if all([not parameters[s] for s in ['reload_character_embeddings', 'reload_character_lstm',
            'reload_token_embeddings', 'reload_token_lstm', 'reload_feedforward', 'reload_crf']]):
            msg = """If use_pretrained_model is set to True, at least one of reload_character_embeddings, 
                reload_character_lstm, reload_token_embeddings, reload_token_lstm, reload_feedforward, 
                reload_crf must be set to True."""
            raise ValueError(msg)

    if parameters['gradient_clipping_value'] < 0:
        parameters['gradient_clipping_value'] = abs(parameters['gradient_clipping_value'])

    if parameters['output_scores'] and parameters['use_crf']:
        warn_msg = """Warning when use_crf is True, scores are decoded
        using the crf. As a result, the scores cannot be directly interpreted
        in terms of class prediction.
        """
        warnings.warn(warn_msg)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

def order_dictionary(dictionary, mode, reverse=False):
    '''
    Order a dictionary by 'key' or 'value'.
    mode should be either 'key' or 'value'
    http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    '''

    if mode =='key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(0),
                                              reverse=reverse))
    elif mode =='value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(1),
                                              reverse=reverse))
    elif mode =='key_value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              reverse=reverse))
    elif mode =='value_key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=lambda x: (x[1], x[0]),
                                              reverse=reverse))
    else:
        raise ValueError("Unknown mode. Should be 'key' or 'value'")

def reverse_dictionary(dictionary):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    http://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain
    '''
    #print('type(dictionary): {0}'.format(type(dictionary)))
    if type(dictionary) is collections.OrderedDict:
        #print(type(dictionary))
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}

def merge_dictionaries(*dict_args):
    '''
    http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def pad_list(old_list, padding_size, padding_value):
    '''
    http://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
    Example: pad_list([6,2,3], 5, 0) returns [6,2,3,0,0]
    '''
    assert padding_size >= len(old_list)
    return old_list + [padding_value] * (padding_size-len(old_list))

def get_basename_without_extension(filepath):
    '''
    Getting the basename of the filepath without the extension
    E.g. 'data/formatted/movie_reviews.pickle' -> 'movie_reviews'
    '''
    return os.path.basename(os.path.splitext(filepath)[0])

def create_folder_if_not_exists(directory):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_current_milliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(int(round(time.time() * 1000)))


def get_current_time_in_seconds():
    '''
    http://stackoverflow.com/questions/415511/how-to-get-current-time-in-python
    '''
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

def get_current_time_in_miliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))


def convert_configparser_to_dictionary(config):
    '''
    http://stackoverflow.com/questions/1773793/convert-configparser-items-to-dictionary
    '''
    my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}
    return my_config_parser_dict

def get_parameter_to_section_of_configparser(config):
    parameter_to_section = {}
    for s in config.sections():
        for p, _ in config.items(s):
            parameter_to_section[p] = s
    return parameter_to_section


def copytree(src, dst, symlinks=False, ignore=None):
    '''
    http://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
    '''
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)