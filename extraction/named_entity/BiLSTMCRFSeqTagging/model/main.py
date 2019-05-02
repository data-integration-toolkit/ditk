"""
@author: Sarvesh Parab
@course: USC CSCI 571
Term : Spring 2019
Email : sparab@usc.edu
Website : http://www.sarveshparab.com
"""

import functools
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

from conll_eval_metrics import evaluate
from model.masked_conv import masked_conv1d_and_max
from model.ner import Ner

from model.eval_metrics import precision, recall, f1

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BiLSTMCRFSeqTag(Ner):

    """
    Setting up the logger for both the application logs and the tensorflow logs
    Application Log path : ../logs/app-{%H_%M_%S}.log
    Tensorflow Log path : ../logs/tf-{%H_%M_%S}.log
    """
    # Setting up the logger
    log = logging.getLogger('root')
    logdatetime = datetime.now().strftime("%H_%M_%S")
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
    log.setLevel(logging.DEBUG)
    logFormatter = logging.Formatter(FORMAT)
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler('../logs/app-'+logdatetime+'.log')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Logging for tensorflow
    Path('../results').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('../logs/tf-'+logdatetime+'.log')
    ]
    logging.getLogger('tensorflow').handlers = handlers

    """
    Default blank constructor
    """
    def __init__(self):
        pass

    """
    Providing definition to the abstract DITK parent class method - convert_ground_truth
    
    # Description: 
        Converts test data into common format for evaluation [i.e. same format as predict()]
        This added step/layer of abstraction is required due to the refactoring of read_dataset_train()
        and read_dataset_test() back to the single method of read_dataset() along with the requirement on
        the format of the output of predict() and therefore the input format requirement of evaluate(). Since
        individuals will implement their own format of data from read_dataset(), this is the layer that
        will convert to proper format for evaluate().
    # Arguments:
        data        - data in proper format for train or test. [i.e. format of output from read_dataset]
        *args       - Not Applicable
        **kwargs    - wordPosition [default : 0] - Column number with the mention word
                    - tagPosition [default : 3] - Column number with the entity tag
                    - writeGroundTruthToFile [default : True] - Flag to enable writing ground truths to a file
                    - groundTruthPath [default : ../results/groundTruths.txt] - Location to save the ground truths file
    # Return:
        [tuple,...], i.e. list of tuples. [SAME format as output of predict()]
            Each tuple is (start index, span, mention text, mention type)
            Where:
             - start index: int, the index of the first character of the mention span. None if not applicable.
             - span: int, the length of the mention. None if not applicable.
             - mention text: str, the actual text that was identified as a named entity. Required.
             - mention type: str, the entity/mention type. None if not applicable.
    
    """
    def convert_ground_truth(self, data, *args, **kwargs):
        self.log.debug("Invoked convert_ground_truth method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        tag_contents = list()
        for i, line in enumerate(data['test']):
            # Check if end of sentence or not
            if len(line) == 0:
                continue
            else:
                tag_contents.append([None, None, line[kwargs.get("wordPosition", 0)], line[kwargs.get("tagPosition", 3)]])

        self.log.debug("Returning ground truths for the test input file :")
        self.log.debug(tag_contents)

        if kwargs.get("writeGroundTruthToFile", True):
            with Path(kwargs.get("groundTruthPath", '../results/groundTruths.txt')).open(mode='w') as f:
                for x in range(len(tag_contents)):
                    line = ""
                    if tag_contents[x][0] is None:
                        line += "-" + " "
                    else:
                        line += tag_contents[x][0] + " "
                    if tag_contents[x][1] is None:
                        line += "-" + " "
                    else:
                        line += tag_contents[x][1] + " "
                    line += tag_contents[x][2] + " "
                    line += tag_contents[x][3]
                    line += "\n"
                    f.write(line)

        return tag_contents

    """
    Providing definition to the abstract DITK parent class method - read_dataset

    # Description: 
        Reads a dataset in preparation for train or test. Returns data in proper format for train or test.
    # Arguments:
        file_dict   - A dictionary with these keys:
                        - train - Location of train dataset file
                        - test - Location of test dataset file
                        - dev - Location of dev dataset file
        dataset_name- Name of the dataset required for calling appropriate utils, converters
        *args       - Not Applicable
        **kwargs    - fileHasHeaders [default : True] - Flag to check if input file has headers
                    - columnDelimiter [default : `space`] - Delimiter in the data input
    # Return:
        A dictionary of file_dict keys as keys and values as lists of lines, where in each line is further tokenized
        on the column delimiter and extracted as a list 

    """
    def read_dataset(self, file_dict, dataset_name="CoNLL03", *args, **kwargs):
        self.log.debug("Invoked read_dataset method")
        self.log.debug("With parameters : ")
        self.log.debug(file_dict)
        self.log.debug(dataset_name)
        self.log.debug(args)
        self.log.debug(kwargs)

        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    if kwargs.get("fileHasHeaders", True):
                        next(f)
                        next(f)
                    raw_data = f.read().splitlines()
                for i, line in enumerate(raw_data):
                    if len(line.strip()) > 0:
                        raw_data[i] = line.strip().split(kwargs.get("columnDelimiter", " "))
                    else:
                        raw_data[i] = list(line)
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)

        self.log.debug("Returning data : ")
        self.log.debug(data)

        return data

    """
    Providing definition to the abstract DITK parent class method - train
    
    # Description: 
        Trains he model on the parsed data
        Calls the internal save_model method to save the trained model for predictions
    # Arguments:
        data        - Parsed input data in the format returned by read_dataset method
        *args       - Not Applicable
        **kwargs    - dimChars [default : 100] - Model character level dimensionality
                    - dim [default : 300] - Model dimensionality
                    - dropout [default : 0.5] - Model dropout rate
                    - epochs [default : 25] - Number of epoch to run for training
                    - batchSize [default : 20] - Training batch sizes
                    - filterNum [default : 50] - Filter area size
                    - lstmSize [default : 100] - State size of the Bi-LSTM layers
                    - vocabWordsPath [default : ../dev/vocab.words.txt] - Location of the parsed words set
                    - vocabCharsPath [default : ../dev/vocab.chars.txt] - Location of the parsed characters set
                    - vocabTagsPath [default : ../dev/vocab.tags.txt] - Location of the parsed tags set
                    - gloveCompressedNPZPath [default : ../dev/glove.npz] - Location of the extracted Glove embeddings from input data in compressed form
                    - paramsPath [default : ../results/params.json] - Location where model parameters get saved
                    - inputFileWordsPath [default : ../dev/{}.words.txt] - Location of the extracted words from the input files
                    - inputFileTagsPath [default : ../dev/{}.tags.txt] - Location of the extracted tags from the input files
                    - checkpointPath [default : ../results/checkpoint] - Location to save intermediate checkpoints
                    - modelPath [default : ../results/saved_model] - Location to save the best model
                    - gloveEmbedPath [default : ../resources/glove/glove.840B.300d.txt] - Location fo the glove embedding file
                    - wordPosition [default : 0] - Column number with the mention word
                    - tagPosition [default : 3] - Column number with the entity tag
                    - writeInputToFile [default : True] - Flag to toggle behaviour of the internal data_converter method
    # Return:
        N/A
    """
    def train(self, data, *args, **kwargs):
        self.log.debug("Invoked train method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        self.data_converter(data, *args, **kwargs)

        params = {
            'dim_chars': kwargs.get("dimChars", 100),
            'dim': kwargs.get("dim", 300),
            'dropout': kwargs.get("dropout", 0.5),
            'num_oov_buckets': 1,
            'epochs': kwargs.get("epochs", 25),
            'batch_size': kwargs.get("batchSize", 20),
            'buffer': 15000,
            'filters': kwargs.get("filterNum", 50),
            'kernel_size': 3,
            'lstm_size': kwargs.get("lstmSize", 100),
            'words': str(kwargs.get("vocabWordsPath", '../dev/vocab.words.txt')),
            'chars': str(kwargs.get("vocabCharsPath", '../dev/vocab.chars.txt')),
            'tags': str(kwargs.get("vocabTagsPath", '../dev/vocab.tags.txt')),
            'glove': str(kwargs.get("gloveCompressedNPZPath", "../dev/glove.npz"))
        }

        with Path(kwargs.get("paramsPath", '../results/params.json')).open('w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

        def fwords(name):
            return str(kwargs.get("inputFileWordsPath", "../dev/" + '{}.words.txt').format(name))

        def ftags(name):
            return str(kwargs.get("inputFileTagsPath", "../dev/" + '{}.tags.txt').format(name))

        # Estimator, train and evaluate
        train_inpf = functools.partial(self.input_fn, fwords('train'), ftags('train'),
                                       params, shuffle_and_repeat=True)
        eval_inpf = functools.partial(self.input_fn, fwords('dev'), ftags('dev'))

        cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
        estimator = tf.estimator.Estimator(self.model_fn, kwargs.get("checkpointPath", "../results/checkpoint"), cfg, params)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        self.log.debug("Training done! ... Saving trained model")

        self.save_model(None, None, **kwargs)

    """
    Providing definition to the abstract DITK parent class method - predict

    # Description:
        Parses and converts the input sentence provided in a file for predicting the NER tags
        Calls the internal load_model method to load the trained model for predictions
    # Arguments:
        data        - The file location with the input text in the common format for prediction
        *args       - Not Applicable
        **kwargs    - fileHasHeaders  [default : True] - Flag to check if input file has headers
                    - loadModelFrom [default : checkpoint] - Flag to decide to choose to load model from 'checkpoint' or 'saved_model'
                    - vocabWordsPath [default : ../dev/vocab.words.txt] - Location of the parsed words set
                    - vocabCharsPath [default : ../dev/vocab.chars.txt] - Location of the parsed characters set
                    - vocabTagsPath [default : ../dev/vocab.tags.txt] - Location of the parsed tags set
                    - gloveCompressedNPZPath [default : ../dev/glove.npz] - Location of the extracted Glove embeddings from input data in compressed form
                    - paramsPath [default : ../results/params.json] - Location where model parameters get saved
                    - checkpointPath [default : ../results/checkpoint] - Location to save intermediate checkpoints
                    - modelPath [default : ../results/saved_model] - Location to save the best model
                    - writePredsToFile [default : True] - Flag to enable writing predictions to file
                    - predsPath [default : ../results/predictions.txt] - Location where to write predictions into
    # Return:
        [tuple,...], i.e. list of tuples.
            Each tuple is (start index, span, mention text, mention type)
            Where:
             - start index: int, the index of the first character of the mention span. None if not applicable.
             - span: int, the length of the mention. None if not applicable.
             - mention text: str, the actual text that was identified as a named entity. Required.
             - mention type: str, the entity/mention type. None if not applicable.

             NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                 evaluation. See note in evaluate() about parallel arrays.]
    """
    def predict(self, data, *args, **kwargs):
        self.log.debug("Invoked predict method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        pred_tuple = list()
        ret_pred_tuple = list()
        raw_data = {}

        with open(data, mode='r', encoding='utf-8') as f:
            if kwargs.get("fileHasHeaders", True):
                next(f)
                next(f)
            file_data = f.read().splitlines()
        for i, line in enumerate(file_data):
            if len(line.strip()) > 0:
                file_data[i] = line.strip().split()
            else:
                file_data[i] = list(line)

        raw_data['test'] = file_data

        word_tag_strings = self.create_word_tag_files(raw_data, **kwargs)
        data_val_string = word_tag_strings[0]
        true_tag_val_string = word_tag_strings[1]

        data_val_list = data_val_string.split(" ")
        true_tag_val_list = true_tag_val_string.split(" ")

        loadedModel = self.load_model(**kwargs)

        if kwargs.get("loadModelFrom", "checkpoint") == "checkpoint":
            predict_inpf = functools.partial(self.predict_input_fn, data_val_string)
            for pred in loadedModel.predict(predict_inpf):
                self.pretty_print(data_val_string, pred['tags'])

                for x in range(min(len(data_val_list), len(true_tag_val_list), len(pred['tags']))):
                    pred_tuple.append([None, None, data_val_list[x], true_tag_val_list[x], pred['tags'][x].decode()])
                    ret_pred_tuple.append([None, None, data_val_list[x], pred['tags'][x].decode()])

                break
        else:
            pred = loadedModel(self.parse_fn_load_model(data_val_string))

            for x in range(min(len(data_val_list), len(true_tag_val_list), len(pred['tags']))):
                pred_tuple.append([None, None, data_val_list[x], true_tag_val_list[x], pred['tags'][x].decode()])
                ret_pred_tuple.append([None, None, data_val_list[x], pred['tags'][x].decode()])

        self.log.debug("Returning predictions :")
        self.log.debug(pred_tuple)

        if kwargs.get("writePredsToFile", True):
            with Path(kwargs.get("predsPath", '../results/predictions.txt')).open(mode='w') as f:
                f.write("WORD TRUE_LABEL PRED_LABEL\n\n")
                for x in range(len(pred_tuple)):
                    f.write(pred_tuple[x][2] + " " + pred_tuple[x][3] + " " + pred_tuple[x][4] + "\n")

        return ret_pred_tuple

    """
    Providing definition to the abstract DITK parent class method - evaluate

    # Description:
        Calculates evaluation metrics on chosen benchmark dataset
            - Precision
            - Recall
            - F1 Score
    # Arguments:
        predictions - List of predicted labels
        groundTruths- List of ground truth labels
        *args       - Not Applicable
        **kwargs    - predsPath [default : ../results/predictions.txt] - Location from where to read predictions from
                    - groundTruthPath [default : ../results/groundTruths.txt] - Location from where to read ground truths from
    # Return:
        Tuple with metrics (p,r,f1). Each element is float.
    """
    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        self.log.debug("Invoked evaluate method")
        self.log.debug("With parameters : ")
        self.log.debug(predictions)
        self.log.debug(groundTruths)
        self.log.debug(args)
        self.log.debug(kwargs)

        true_vals = list()
        pred_vals = list()

        if predictions is None and groundTruths is None:
            with open(kwargs.get("predsPath", '../results/predictions.txt'), mode='r', encoding='utf-8') as f:
                next(f)
                next(f)
                raw_preds = f.read().splitlines()

            for x in range(len(raw_preds)):
                true_vals.append(raw_preds[x].split(" ")[1])
                pred_vals.append(raw_preds[x].split(" ")[2])

        else:
            true_vals = groundTruths
            pred_vals = predictions

        eval_metrics = evaluate(true_vals, pred_vals, False)

        self.log.debug("Returning evaluation metrics [P, R, F1] :")
        self.log.debug(eval_metrics)

        return eval_metrics

    def save_model(self, file=None, *args, **kwargs):
        self.log.debug("Invoked save_model method")
        self.log.debug("With parameters : ")
        self.log.debug(kwargs.get("modelPath", "../results/saved_model"))
        self.log.debug(args)
        self.log.debug(kwargs)

        with Path(kwargs.get("paramsPath", '../results/params.json')).open() as f:
            params = json.load(f)

        params['words'] = str(kwargs.get("vocabWordsPath", '../dev/vocab.words.txt'))
        params['chars'] = str(kwargs.get("vocabCharsPath", '../dev/vocab.chars.txt'))
        params['tags'] = str(kwargs.get("vocabTagsPath", '../dev/vocab.tags.txt'))
        params['glove'] = str(kwargs.get("gloveCompressedNPZPath", "../dev/glove.npz"))

        estimator = tf.estimator.Estimator(self.model_fn, kwargs.get("checkpointPath", "../results/checkpoint"), params=params)
        estimator.export_saved_model(kwargs.get("modelPath", "../results/saved_model"), self.serving_input_receiver_fn)

        self.log.debug("Model saved at location : %s", kwargs.get("modelPath", "../results/saved_model"))

    def load_model(self, file=None, **kwargs):
        self.log.debug("Invoked load_model method")
        self.log.debug("With parameters : ")
        self.log.debug(kwargs)

        if kwargs.get("loadModelFrom", "checkpoint") == "checkpoint":
            return self.load_model_from_checkpoint(**kwargs)
        else:
            return self.load_model_from_saved_model(**kwargs)

    def load_model_from_checkpoint(self, **kwargs):
        with Path(kwargs.get("paramsPath", '../results/params.json')).open() as f:
            params = json.load(f)

        params['words'] = str(kwargs.get("vocabWordsPath", '../dev/vocab.words.txt'))
        params['chars'] = str(kwargs.get("vocabCharsPath", '../dev/vocab.chars.txt'))
        params['tags'] = str(kwargs.get("vocabTagsPath", '../dev/vocab.tags.txt'))
        params['glove'] = str(kwargs.get("gloveCompressedNPZPath", "../dev/glove.npz"))

        estimator = tf.estimator.Estimator(self.model_fn, kwargs.get("checkpointPath", "../results/checkpoint"), params=params)

        self.log.debug("Loaded model from previous checkpoint location : %s", kwargs.get("checkpointPath", "../results/checkpoint"))
        return estimator

    def load_model_from_saved_model(self, **kwargs):

        subdirs = [x for x in Path(kwargs.get("modelPath", "../results/saved_model")).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        # predictions = predict_fn(parse_fn(LINE))

        self.log.debug("Loaded model from saved model at location : %s", kwargs.get("modelPath", "../results/saved_model"))
        return predict_fn

    def data_converter(self, data, *args, **kwargs):
        self.log.debug("Invoked data_converter method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        self.log.debug("Parsing input and building the corresponding data files")
        self.create_word_tag_files(data, **kwargs)

        self.log.debug("Building vocab files")
        self.build_vocab_files(**kwargs)

        self.log.debug("Fetching glove embeddings from file : %s",
                       kwargs.get("gloveEmbedPath", "../resources/glove/glove.840B.300d.txt"))
        self.incorporate_glove_embeddings(**kwargs)

    def create_word_tag_files(self, data, **kwargs):
        def words(name):
            return kwargs.get("inputFileWordsPath", "../dev/" + '{}.words.txt').format(name)

        def tags(name):
            return kwargs.get("inputFileTagsPath", "../dev/" + '{}.tags.txt').format(name)

        words_string = ''
        tags_string = ''

        for file_type in data.keys():
            words_contents = ''
            tag_contents = ''
            for i, val in enumerate(data[file_type]):
                # Check if end of sentence or not
                if len(val) == 0:
                    words_contents.strip()
                    tag_contents.strip()
                    words_contents += '\n'
                    tag_contents += '\n'
                else:
                    words_contents += val[kwargs.get("wordPosition", 0)] + " "
                    tag_contents += val[kwargs.get("tagPosition", 3)] + " "

            if kwargs.get("writeInputToFile", True):
                with Path(words(file_type)).open(mode='w') as f:
                    f.write(words_contents.strip())
                with Path(tags(file_type)).open(mode='w') as f:
                    f.write(tag_contents.strip())
            else:
                words_string += words_contents.strip()
                tags_string += tag_contents.strip()

        if not kwargs.get("writeInputToFile", True):
            return [words_string.replace('\n', '').replace('\r', ''), tags_string.replace('\n', '').replace('\r', '')]

    def build_vocab_files(self, **kwargs):
        MINCOUNT = 1

        # 1. Words
        # Get Counter of words on all the data, filter by min count, save
        def words(name):
            return kwargs.get("inputFileWordsPath", "../dev/" + '{}.words.txt').format(name)

        self.log.debug('Build vocab words (may take a while)')
        counter_words = Counter()
        for n in ['train', 'test', 'dev']:
            with Path(words(n)).open() as f:
                for line in f:
                    counter_words.update(line.strip().split())
        vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}
        with Path(kwargs.get("vocabWordsPath", '../dev/vocab.words.txt')).open('w') as f:
            for w in sorted(list(vocab_words)):
                f.write('{}\n'.format(w))
        self.log.debug('- done. Kept {} out of {}'.format(
            len(vocab_words), len(counter_words)))
        # 2. Chars
        # Get all the characters from the vocab words
        self.log.debug('Build vocab chars')
        vocab_chars = set()
        for w in vocab_words:
            vocab_chars.update(w)
        with Path(kwargs.get("vocabCharsPath", '../dev/vocab.chars.txt')).open('w') as f:
            for c in sorted(list(vocab_chars)):
                f.write('{}\n'.format(c))
        self.log.debug('- done. Found {} chars'.format(len(vocab_chars)))

        # 3. Tags
        # Get all tags from the training set
        def tags(name):
            return kwargs.get("inputFileTagsPath", "../dev/" + '{}.tags.txt').format(name)

        self.log.debug('Build vocab tags (may take a while)')
        vocab_tags = set()
        with Path(tags('train')).open() as f:
            for line in f:
                vocab_tags.update(line.strip().split())
        with Path(kwargs.get("vocabTagsPath", '../dev/vocab.tags.txt')).open('w') as f:
            for t in sorted(list(vocab_tags)):
                f.write('{}\n'.format(t))
        self.log.debug('- done. Found {} tags.'.format(len(vocab_tags)))

    def incorporate_glove_embeddings(self, **kwargs):
        # Load vocab
        with Path(kwargs.get("vocabWordsPath", '../dev/vocab.words.txt')).open() as f:
            word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
        size_vocab = len(word_to_idx)
        # Array of zeros
        embeddings = np.zeros((size_vocab, 300))
        # Get relevant glove vectors
        found = 0
        self.log.debug('Reading GloVe file (may take a while)')
        with Path(kwargs.get("gloveEmbedPath", "../resources/glove/glove.840B.300d.txt")).open(mode='r', encoding='utf8') as f:
            for line_idx, line in enumerate(f):
                if line_idx % 100000 == 0:
                    self.log.debug('- At line {}'.format(line_idx))
                line = line.strip().split()
                if len(line) != 300 + 1:
                    continue
                word = line[0]
                embedding = line[1:]
                if word in word_to_idx:
                    found += 1
                    word_idx = word_to_idx[word]
                    embeddings[word_idx] = embedding
        self.log.debug('- done. Found {} vectors for {} words'.format(found, size_vocab))
        # Save np.array to file
        np.savez_compressed(kwargs.get("gloveCompressedNPZPath", "../dev/glove.npz"), embeddings=embeddings)

    def input_fn(self, words, tags, params=None, shuffle_and_repeat=False):
        params = params if params is not None else {}
        shapes = ((([None], ()),               # (words, nwords)
                   ([None, None], [None])),    # (chars, nchars)
                  [None])                      # tags
        types = (((tf.string, tf.int32),
                  (tf.string, tf.int32)),
                 tf.string)
        defaults = ((('<pad>', 0),
                     ('<pad>', 0)),
                    'O')
        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.generator_fn, words, tags),
            output_shapes=shapes, output_types=types)

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

        dataset = (dataset
                   .padded_batch(params.get('batch_size', 20), shapes, defaults)
                   .prefetch(1))
        return dataset

    def generator_fn(self, words, tags):
        with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
            for line_words, line_tags in zip(f_words, f_tags):
                yield self.parse_fn(line_words, line_tags)

    def parse_fn(self, line_words, line_tags):
        # Encode in Bytes for TF
        words = [w.encode() for w in line_words.strip().split()]
        tags = [t.encode() for t in line_tags.strip().split()]
        assert len(words) == len(tags), "Words and tags lengths don't match"

        # Chars
        chars = [[c.encode() for c in w] for w in line_words.strip().split()]
        lengths = [len(c) for c in chars]
        max_len = max(lengths)
        chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
        return ((words, len(words)), (chars, lengths)), tags

    def model_fn(self, features, labels, mode, params):
        # For serving features are a bit different
        if isinstance(features, dict):
            features = ((features['words'], features['nwords']),
                        (features['chars'], features['nchars']))

        # Read vocabs and inputs
        dropout = params['dropout']
        (words, nwords), (chars, nchars) = features
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        vocab_words = tf.contrib.lookup.index_table_from_file(
            params['words'], num_oov_buckets=params['num_oov_buckets'])
        vocab_chars = tf.contrib.lookup.index_table_from_file(
            params['chars'], num_oov_buckets=params['num_oov_buckets'])
        with Path(params['tags']).open() as f:
            indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
            num_tags = len(indices) + 1
        with Path(params['chars']).open() as f:
            num_chars = sum(1 for _ in f) + params['num_oov_buckets']

        # Char Embeddings
        char_ids = vocab_chars.lookup(chars)
        variable = tf.get_variable(
            'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
        char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
        char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                            training=training)

        # Char 1d convolution
        weights = tf.sequence_mask(nchars)
        char_embeddings = masked_conv1d_and_max(
            char_embeddings, weights, params['filters'], params['kernel_size'])

        # Word Embeddings
        word_ids = vocab_words.lookup(words)
        glove = np.load(params['glove'])['embeddings']  # np.array
        variable = np.vstack([glove, [[0.] * params['dim']]])
        variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
        word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

        # Concatenate Word and Char Embeddings
        embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
        embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

        # LSTM
        t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.layers.dropout(output, rate=dropout, training=training)

        # CRF
        logits = tf.layers.dense(output, num_tags)
        crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
        pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Predictions
            reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
                params['tags'])
            pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
            predictions = {
                'pred_ids': pred_ids,
                'tags': pred_strings
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            # Loss
            vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
            tags = vocab_tags.lookup(labels)
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                logits, tags, nwords, crf_params)
            loss = tf.reduce_mean(-log_likelihood)

            # Metrics
            weights = tf.sequence_mask(nwords)
            metrics = {
                'acc': tf.metrics.accuracy(tags, pred_ids, weights),
                'precision': precision(tags, pred_ids, num_tags, indices, weights),
                'recall': recall(tags, pred_ids, num_tags, indices, weights),
                'f1': f1(tags, pred_ids, num_tags, indices, weights),
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            elif mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.train.AdamOptimizer().minimize(
                    loss, global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    def serving_input_receiver_fn(self):
        """Serving input_fn that builds features from placeholders

        Returns
        -------
        tf.estimator.export.ServingInputReceiver
        """
        words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
        nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
        chars = tf.placeholder(dtype=tf.string, shape=[None, None, None],
                               name='chars')
        nchars = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                name='nchars')
        receiver_tensors = {'words': words, 'nwords': nwords,
                            'chars': chars, 'nchars': nchars}
        features = {'words': words, 'nwords': nwords,
                    'chars': chars, 'nchars': nchars}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    def predict_input_fn(self, line):
        # Words
        words = [w.encode() for w in line.strip().split()]
        nwords = len(words)

        # Chars
        chars = [[c.encode() for c in w] for w in line.strip().split()]
        lengths = [len(c) for c in chars]
        max_len = max(lengths)
        chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

        # Wrapping in Tensors
        words = tf.constant([words], dtype=tf.string)
        nwords = tf.constant([nwords], dtype=tf.int32)
        chars = tf.constant([chars], dtype=tf.string)
        nchars = tf.constant([lengths], dtype=tf.int32)

        return ((words, nwords), (chars, nchars)), None

    def pretty_print(self, line, preds):
        words = line.strip().split()
        lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
        padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
        padded_preds = [p.decode() + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
        self.log.debug('words: {}'.format(' '.join(padded_words)))
        self.log.debug('preds: {}'.format(' '.join(padded_preds)))

    def parse_fn_load_model(self, line):
        # Encode in Bytes for TF
        words = [w.encode() for w in line.strip().split()]

        # Chars
        chars = [[c.encode() for c in w] for w in line.strip().split()]
        lengths = [len(c) for c in chars]
        max_len = max(lengths)
        chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

        return {'words': [words], 'nwords': [len(words)],
                'chars': [chars], 'nchars': [lengths]}

    def main(self, input_file, **kwargs):

        file_dict = dict()
        file_dict['train'] = input_file
        file_dict['test'] = input_file
        file_dict['dev'] = input_file

        data = self.read_dataset(file_dict, "CoNLL2003", None, **kwargs)
        groundTruth = self.convert_ground_truth(data, None, **kwargs)
        self.train(data, None, **kwargs)
        predictions = self.predict(input_file, None, writeInputToFile=False, **kwargs)
        self.evaluate([col[3] for col in predictions], [col[3] for col in groundTruth], None, **kwargs)

        return "../results/predictions.txt"
