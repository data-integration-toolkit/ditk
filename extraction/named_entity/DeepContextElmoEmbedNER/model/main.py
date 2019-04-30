"""
@author: Sarvesh Parab
@course: USC CSCI 571
Term : Spring 2019
Email : sparab@usc.edu
Website : http://www.sarveshparab.com
"""

import os
from datetime import datetime
import numpy as np
import pickle
import logging
import sys
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from model.Elmo import ElmoModel
from utils.LSTMCNNCRFeeder import LSTMCNNCRFeeder
from utils.conlleval import evaluate
from utils.checkmate import BestCheckpointSaver, best_checkpoint

from model.bilm.data import Batcher
from model.bilm.model import BidirectionalLanguageModel
from model.ner import Ner

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class DeepElmoEmbedNer(Ner):
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
    fileHandler = logging.FileHandler('../logs/app-' + logdatetime + '.log')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Logging for tensorflow
    Path('../results').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('../logs/tf-' + logdatetime + '.log'),
        logging.StreamHandler(sys.stdout)
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
                tag_contents.append(
                    [None, None, line[kwargs.get("wordPosition", 0)], line[kwargs.get("tagPosition", 3)]])

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
    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
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
        return data
    """
    Providing definition to the abstract DITK parent class method - train
    
    # Description: 
        Trains he model on the parsed data
        Calls the internal save_model method to save the trained model for predictions
    # Arguments:
        data        - Parsed input data in the format returned by read_dataset method
        *args       - Not Applicable
        **kwargs    - parsedDumpPath [default : ../dev/parsedDataDump.pkl] - Location of the parsed input data-files in the pickled format
                    - vocabPath [default : ../dev/vocab.txt] - Location of the parsed vocab
                    - elmoOptionsFile [default : ../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json] - ELMo model options parameters file
                    - elmoWeightFile [default : ../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5] - ELMo model weights file 
                    - wordEmbeddingSize [default : 50] - Set the ELMo word embedding size for the model
                    - charEmbeddingSize [default : 16] - Set the ELMo character embedding size for the model
                    - LSTMStateSize [default : 200] - State size of the Multi-LSTM layers
                    - filterNum [default : 128] - Filter area size
                    - filterSize [default : 3] - Number of filters in the model
                    - learningRate [default : 0.015] - Model learning rate
                    - dropoutRate [default : 0.5] - Model dropout rate
                    - epochWidth [default : 16] - Batch size within each epoch
                    - maxEpoch [default : 100] - Number of epoch to run for training
                    - checkpointPath [default : ../results/checkpoints] - Location to save intermediate checkpoints
                    - bestCheckpointPath [default : ../results/checkpoints/best] - Location to save the best F1 returning 
                    - trainWordsPath [default : ../dev/train.word.vocab] - Location to save the intermediate vocabulary words from the training set
                    - trainCharPath [default : ../dev/train.char.vocab] - Location to save the intermediate vocabulary characters from the training set
                    - gloveEmbedPath [default : ../resources/glove/glove.6B.50d.txt] - Location fo the glove embedding file
                    - fetchPredictData [default : False] - Flag to toggle behaviour of the internal data_converter method
                    - maxWordLength [default : 30] - Set maximal word length for the model
                    - wordPosition [default : 0] - Column number with the mention word
                    - tagPosition [default : 3] - Column number with the entity tag
    # Return:
        model   - Trained ELMo model object
        sess    - Tensorflow session object (will be used to maintain the tensorflow session instance)
        saver   - Tensorflow saver instance (will be used to load model again)
    
    """
    def train(self, data, *args, **kwargs):

        if not os.path.isfile(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')):
            self.data_converter(data, *args, **kwargs)

        with open(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl'), 'rb') as fp:
            train_set, val_set, test_set, dicts = pickle.load(fp)

        w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
        idx2w = {w2idx[k]: k for k in w2idx}
        idx2la = {la2idx[k]: k for k in la2idx}

        train_x, train_chars, train_la = train_set
        val_x, val_chars, val_la = val_set
        test_x, test_chars, test_la = test_set

        self.log.debug('Loading elmo!')
        elmo_batcher = Batcher(kwargs.get("vocabPath", '../dev/vocab.txt'), 50)
        elmo_bilm = BidirectionalLanguageModel(kwargs.get("elmoOptionsFile", '../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'),
                                               kwargs.get("elmoWeightFile", '../resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'))

        self.log.debug('Loading model!')

        num_classes = len(la2idx.keys())
        max_seq_length = max(
            max(map(len, train_x)),
            max(map(len, test_x)),
        )
        max_word_length = max(
            max([len(ssc) for sc in train_chars for ssc in sc]),
            max([len(ssc) for sc in test_chars for ssc in sc])
        )

        model = ElmoModel(
            True,
            kwargs.get("wordEmbeddingSize", 50),  # Word embedding size
            kwargs.get("charEmbeddingSize", 16),  # Character embedding size
            kwargs.get("LSTMStateSize", 200),  # LSTM state size
            kwargs.get("filterNum", 128),  # Filter num
            kwargs.get("filterSize", 3),  # Filter size
            num_classes,
            max_seq_length,
            max_word_length,
            kwargs.get("learningRate", 0.015),
            kwargs.get("dropoutRate", 0.5),
            elmo_bilm,
            1,  # elmo_mode
            elmo_batcher,
            **kwargs)

        self.log.debug('Start training...')
        self.log.debug('Train size = %d' % len(train_x))
        self.log.debug('Val size = %d' % len(val_x))
        self.log.debug('Test size = %d' % len(test_x))
        self.log.debug('Num classes = %d' % num_classes)

        start_epoch = 1
        max_epoch = kwargs.get("maxEpoch", 100)

        self.log.debug('Epoch = %d' % max_epoch)

        saver = tf.train.Saver()
        best_saver = BestCheckpointSaver(
            save_dir=kwargs.get("bestCheckpointPath", "../results/checkpoints/best"),
            num_to_keep=1,
            maximize=True
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=kwargs.get("checkpointPath", "../results/checkpoints"))
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        train_feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, kwargs.get("epochWidth", 16))
        val_feeder = LSTMCNNCRFeeder(val_x, val_chars, val_la, max_seq_length, max_word_length, kwargs.get("epochWidth", 16))

        for epoch in range(start_epoch, max_epoch + 1):
            loss = 0
            for step in range(train_feeder.step_per_epoch):
                tokens, chars, labels = train_feeder.feed()

                step_loss = model.train_step(sess, tokens, chars, labels)
                loss += step_loss

                self.log.debug('epoch: %d, size: %d/%d, step_loss: %f, epoch_loss: %f',
                             epoch, train_feeder.offset, train_feeder.size, step_loss, loss)

            preds = []
            for step in range(val_feeder.step_per_epoch):
                tokens, chars, labels = val_feeder.feed()
                pred = model.test(sess, tokens, chars)
                preds.extend(pred)
            true_seqs = [idx2la[la] for sl in val_la for la in sl]
            pred_seqs = [idx2la[la] for sl in preds for la in sl]
            ll = min(len(true_seqs), len(pred_seqs))

            self.log.debug(true_seqs[:ll])
            self.log.debug(pred_seqs[:ll])

            prec, rec, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

            self.log.debug("Epoch: %d, val_p: %f, val_r: %f, val_f1: %f", epoch, prec, rec, f1)

            val_feeder.next_epoch(False)

            saver.save(sess, kwargs.get("checkpointPath", "../results/checkpoints") + '/model.ckpt', global_step=epoch)
            best_saver.handle(f1, sess, epoch)

            logging.info('')
            train_feeder.next_epoch()

        self.log.debug("Training done! ... Saving trained model")
        return model, sess, saver

    """
    Providing definition to the abstract DITK parent class method - predict
    
    # Description:
        Predicts on the given input data. Assumes model has been trained with train()
        Calls the internal load_model method to load the trained model for predictions
    # Arguments:
        data        - The file location with the input text in the common format for prediction
        *args       - Not Applicable
        **kwargs    - model [default : N/A] - ElmoModel instance to hold the loaded model into
                    - sess [default : N/A] - Tensorflow.Session instance used to maintain the same session used to train
                    - saver [default : N/A] - Tensorflow.train.saver instance used to load the trained model
                    - trainedData [default : N/A] - Parsed trained data
                    - fileHasHeaders  [default : True] - Flag to check if input file has headers
                    - parsedDumpPath [default : ../dev/parsedDataDump.pkl] - Location of the parsed input data-files in the pickled format
                    - bestCheckpointPath [default : ../results/checkpoints/best] - Location to save the best F1 returning
                    - epochWidth [default : 16] - Batch size within each epoch
                    - writePredsToFile [default : True] - Flag to enable writing predictions to file
                    - predsPath [default : ../results/predictions.txt] - Location where to write predictions into
                    - writeInputToFile [Default : False] - Flag to toggle behaviour of the internal data_converter method
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

        with open(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl'), 'rb') as fp:
            _, _, _, dicts = pickle.load(fp)
        w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
        idx2la = {la2idx[k]: k for k in la2idx}

        sess = kwargs.get("sess", "")
        model = kwargs.get("model", "")
        saver = kwargs.get("saver", "")

        best_saved_cp = best_checkpoint(kwargs.get("bestCheckpointPath", "../results/checkpoints/best"), True)
        saver.restore(sess, best_saved_cp)

        # Load the model and TF session
        feeder = self.load_model(None, predictData=data, loadForPredict=True, **kwargs)

        # Fetching the predictions
        for _ in tqdm(range(feeder.step_per_epoch)):
            tokens, chars, labels = feeder.feed()

            out = model.decode(sess, tokens, chars, 1)
            for i, preds in enumerate(out):
                length = len(preds[0])

                st = tokens[i, :length].tolist()
                sl = [idx2la[la] for la in labels[i, :length].tolist()]

                preds = [[idx2la[la] for la in pred] for pred in preds]

                for zipped_res in zip(*[st, sl, *preds]):
                    pred_tuple.append([zipped_res[0], zipped_res[1], zipped_res[2]])
                    ret_pred_tuple.append([None, None, zipped_res[0], zipped_res[2]])

                pred_tuple.append([None, None, None])

        self.log.debug("Returning predictions :")
        self.log.debug(pred_tuple)

        if kwargs.get("writePredsToFile", True):
            with Path(kwargs.get("predsPath", '../results/predictions.txt')).open(mode='w') as f:
                f.write("WORD TRUE_LABEL PRED_LABEL\n\n")
                for x in range(len(pred_tuple)):
                    if pred_tuple[x][0] is None or pred_tuple[x][1] is None or pred_tuple[x][2] is None:
                        f.write("\n")
                    else:
                        f.write(pred_tuple[x][0] + " " + pred_tuple[x][1] + " " + pred_tuple[x][2] + "\n")

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
        **kwargs    - predsPath [default : ../results/predictions.txt] - Location where to write predictions into
                    - groundTruthPath [default : ../results/groundTruths.txt] - Location to save the ground truths file
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
                if raw_preds[x] != "" or len(raw_preds[x]) != 0:
                    true_vals.append(raw_preds[x].split(" ")[1])
                    pred_vals.append(raw_preds[x].split(" ")[2])

        else:
            true_vals = groundTruths
            pred_vals = predictions

        eval_metrics = evaluate(true_vals, pred_vals, False)

        self.log.debug("Returning evaluation metrics [P, R, F1] :")
        self.log.debug(eval_metrics)

        return eval_metrics

    def save_model(self, file=None, **kwargs):
        pass

    def load_model(self, file=None, *args, **kwargs):
        self.log.debug("Invoked load_model method")
        self.log.debug("With parameters : ")
        self.log.debug(file)
        self.log.debug(args)
        self.log.debug(kwargs)

        with open(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl'), 'rb') as fp:
            train_set, val_set, test_set, dicts = pickle.load(fp)

        if kwargs.get("loadForPredict", False):
            raw_data = {}
            with open(kwargs.get("predictData", None), mode='r', encoding='utf-8') as f:
                if kwargs.get("fileHasHeaders", True):
                    next(f)
                    next(f)
                file_data = f.read().splitlines()
            for i, line in enumerate(file_data):
                if len(line.strip()) > 0:
                    file_data[i] = line.strip().split()
                else:
                    file_data[i] = list(line)

            file_data.append(list(""))

            raw_data['test'] = file_data
            raw_data['train'] = kwargs.get("trainedData", "")

            test_x, test_chars, test_la = self.data_converter(raw_data, None, fetchPredictData=True, **kwargs)
        else:
            test_x, test_chars, test_la = test_set

        train_x, train_chars, train_la = train_set
        val_x, val_chars, val_la = val_set

        max_seq_length = max(
            max(map(len, train_x)),
            max(map(len, test_x)),
        )
        max_word_length = max(
            max([len(ssc) for sc in train_chars for ssc in sc]),
            max([len(ssc) for sc in test_chars for ssc in sc])
        )

        test_feeder = LSTMCNNCRFeeder(test_x, test_chars, test_la, max_seq_length, max_word_length, kwargs.get("epochWidth", 16))

        if kwargs.get("loadForPredict", False):
            return test_feeder
        else:
            return None

    def data_converter(self, data, *args, **kwargs):
        self.log.debug("Invoked data_converter method")
        self.log.debug("With parameters : ")
        self.log.debug(data)
        self.log.debug(args)
        self.log.debug(kwargs)

        word_set = set()
        char_set = set()
        label_set = set()
        vocab = set()
        net_dump = []

        max_word_len = kwargs.get("maxWordLength", 30)

        # Iterate over each file in dictionary
        for file_type in data.keys():
            file_sentence = []
            file_chars = []
            file_labels = []

            line_sentence = []
            line_chars = []
            line_labels = []
            for i, line in enumerate(data[file_type]):
                # Check if end of sentence
                if len(line) == 0 or i == len(data[file_type]):
                    file_sentence.append(line_sentence)
                    file_chars.append(line_chars)
                    file_labels.append(line_labels)

                    line_sentence = []
                    line_chars = []
                    line_labels = []
                else:
                    word = line[kwargs.get("wordPosition", 0)]
                    chars = [ch for ch in word]
                    label = line[kwargs.get("tagPosition", 3)]

                    if len(chars) > max_word_len:
                        chars = chars[:max_word_len]

                    line_sentence.append(word)
                    line_chars.append(chars)
                    line_labels.append(label)

                    # Should only update word in train set
                    if file_type == 'train':
                        word_set.add(word.lower())
                        char_set.update(*chars)
                        label_set.add(label)

                    vocab.add(word)

            net_dump.append([file_sentence, file_chars, file_labels])

        if kwargs.get("fetchPredictData", False):
            labels2idx = {}
            for idx, label in enumerate(sorted(label_set)):
                labels2idx[label] = idx

            net_dump[0][2] = [np.array([labels2idx[la] for la in sl]) for sl in net_dump[0][2]]  # label

            return net_dump[0]

        words2idx = {}
        chars2idx = {}
        labels2idx = {}

        with Path(kwargs.get("trainWordsPath", '../dev/train.word.vocab')).open(mode='w') as f:
            for idx, word in enumerate(sorted(word_set)):
                words2idx[word] = idx
                f.write(word + '\n')

        with Path(kwargs.get("vocabPath", '../dev/vocab.txt')).open(mode='w', encoding='gb18030') as f:
            vocab = sorted(vocab)
            vocab.insert(0, '<S>')
            vocab.insert(1, '</S>')
            vocab.insert(2, '<UNK>')
            for word in vocab:
                f.write(word + '\n')

        with Path(kwargs.get("trainCharPath", '../dev/train.char.vocab')).open(mode='w') as f:
            for idx, char in enumerate(sorted(char_set)):
                chars2idx[char] = idx
                f.write(char + '\n')

        for idx, label in enumerate(sorted(label_set)):
            labels2idx[label] = idx

        for i in range(len(data.keys())):
            net_dump[i][2] = [np.array([labels2idx[la] for la in sl]) for sl in net_dump[i][2]]  # label

        if len(data.keys()) == 3:
            with Path(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')).open(mode='wb') as f:
                pickle.dump((net_dump[0], net_dump[1], net_dump[2],
                             {
                                 'words2idx': words2idx,
                                 'chars2idx': chars2idx,
                                 'labels2idx': labels2idx
                             }), f)
        elif len(data.keys()) == 2:
            with Path(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')).open(mode='wb') as f:
                pickle.dump((net_dump[0], net_dump[1],
                             {
                                 'words2idx': words2idx,
                                 'chars2idx': chars2idx,
                                 'labels2idx': labels2idx
                             }), f)
        elif len(data.keys()) == 1:
            with Path(kwargs.get("parsedDumpPath", '../dev/parsedDataDump.pkl')).open(mode='wb') as f:
                pickle.dump((net_dump[0],
                             {
                                 'words2idx': words2idx,
                                 'chars2idx': chars2idx,
                                 'labels2idx': labels2idx
                             }), f)

        self.log.debug("Data conversion done!!")

    def main(self, input_file, **kwargs):

        file_dict = dict()
        file_dict['train'] = input_file
        file_dict['test'] = input_file
        file_dict['dev'] = input_file

        data = self.read_dataset(file_dict, "CoNLL2003", None, **kwargs)
        groundTruth = self.convert_ground_truth(data, None, **kwargs)
        model, sess, saver = self.train(data, None, maxEpoch=1, **kwargs)
        predictions = self.predict(input_file, None, writeInputToFile=False, model=model, sess=sess, saver=saver, trainedData=data['train'], **kwargs)
        self.evaluate([col[3] for col in predictions], [col[3] for col in groundTruth], None, **kwargs)

        return "../results/predictions.txt"