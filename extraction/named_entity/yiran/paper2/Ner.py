import codecs
import os
from collections import defaultdict

import keras
import numpy
from keras_wc_embd import get_batch_input, get_dicts_generator, get_embedding_weights_from_file

from extraction.named_entity.ner import Ner
from extraction.named_entity.yiran.paper2.model import build_model


class UOI(Ner):
    MODEL_PATH = 'model.h5'
    WORD_EMBD_PATH = 'dataset/glove.6B.100d.txt'
    RNN_NUM = 16
    RNN_UNITS = 32

    BATCH_SIZE = 16
    EPOCHS = 1
    TAGS = {
        'O': 0,
        'B-PER': 1,
        'I-PER': 2,
        'B-LOC': 3,
        'I-LOC': 4,
        'B-ORG': 5,
        'I-ORG': 6,
        'B-MISC': 7,
        'I-MISC': 8,
    }

    train_taggings = []
    train_sentences = []
    word_embd_weights = None
    word_dict = None
    char_dict = None
    max_word_len = None
    valid_sentences = []
    valid_taggings = []
    valid_steps = None
    model = None
    tag_map = defaultdict()
    total_pred, total_true, matched_num = 0, 0, 0.0
    '''
    The paper is for using Parallel Recurrent Neural Networks to do the NER
    '''

    def evaluate(self, predictions=None, ground_truths=None, *args, **kwargs):
        eps = 1e-6
        precision = (self.matched_num + eps) / (self.total_pred + eps)
        recall = (self.matched_num + eps) / (self.total_true + eps)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def convert_ground_truth(self, data, *args, **kwargs):
        """
        This method will return the ground truth of the words. 
        The data must be from training dataset e.g. connl2003/train.txt, 
        testing dataset e.g. connl2003/valid.txt or dev dataset e.g.connl2003/test.txt
        :param data: a list of words
        :return: a list of tags for the words
        :raise: if the data contains a token having no tag
        """
        result = []
        for word in data:
            result.append(self.tag_map[word])
        return result

    def read_dataset(self, input_files, embedding, *args, **kwargs):
        """
        The method is for reading the data from files.

        :param input_files: An array containing the pathes for training file and validation file. input_files[0] is training data set and input_files[1] is validating file
         For example 
        ["/Users/liyiran/csci548sp19projectner_my/paper2/CoNNL2003eng/train.txt",'/Users/liyiran/csci548sp19projectner_my/paper2/CoNNL2003eng/valid.txt']
        :param embedding the path of embedding file
        for example:
        /Users/liyiran/ditk/extraction/named_entity/yiran/embedding/glove.6B.100d.txt
        :return: training sentences/ tags and validating sentences/ tags
        :raise: if the input is not an array of length 2 or the files do not exists or 
        """
        if not len(input_files) is 2:
            raise NameError('input files should contains two elements: training file, validation')
        self.WORD_EMBD_PATH = embedding
        training_file = input_files[0]
        validate_file = input_files[1]
        sentences, taggings = self.load_data(training_file)
        self.train_sentences.extend(sentences)
        self.train_taggings.extend(taggings)
        sentences, taggings = self.load_data(validate_file)
        self.valid_sentences.extend(sentences)
        self.valid_taggings.extend(taggings)
        return self.train_sentences, self.train_taggings, self.valid_sentences, self.valid_taggings

    def load_data(self, path):
        sentences, taggings = [], []
        with codecs.open(path, 'r', 'utf8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    if not sentences or len(sentences[-1]) > 0:
                        sentences.append([])
                        taggings.append([])
                    continue
                parts = line.split()
                if parts[0] != '-DOCSTART-':
                    self.tag_map[parts[0]] = parts[-1]
                    sentences[-1].append(parts[0])
                    taggings[-1].append(self.TAGS[parts[-1]])
        if not sentences[-1]:
            sentences.pop()
            taggings.pop()
        return sentences, taggings

    # @classmethod
    def train(self, data=None, *args, **kwargs):
        """
        This method is for training the cnn model. After training procedure, the model will be saved in model.h5 file
        :param data: is not used in this method since the training and validating files has been given in read_dataset() method
        :return: None
        """
        dicts_generator = get_dicts_generator(
            word_min_freq=2,
            char_min_freq=2,
            word_ignore_case=True,
            char_ignore_case=False
        )
        for sentence in self.train_sentences:
            dicts_generator(sentence)
        self.word_dict, self.char_dict, self.max_word_len = dicts_generator(return_dict=True)
        if os.path.exists(self.WORD_EMBD_PATH):
            print('Embedding...')
            self.word_dict = {
                '': 0,
                '<UNK>': 1,
            }
            with codecs.open(self.WORD_EMBD_PATH, 'r', 'utf8') as reader:
                print('Embedding open file')
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    word = line.split()[0].lower()
                    if word not in self.word_dict:
                        self.word_dict[word] = len(self.word_dict)
                print('Embedding for loop')
            self.word_embd_weights = get_embedding_weights_from_file(
                self.word_dict,
                self.WORD_EMBD_PATH,
                ignore_case=True,
            )
            print('Embedding done')
        else:
            self.word_embd_weights = None
            raise NameError('embedding file is not found')
        print('Embedding all done')
        train_steps = (len(self.train_sentences) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        valid_steps = (len(self.valid_sentences) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        self.model = build_model(rnn_num=self.RNN_NUM,
                                 rnn_units=self.RNN_UNITS,
                                 word_dict_len=len(self.word_dict),
                                 char_dict_len=len(self.char_dict),
                                 max_word_len=self.max_word_len,
                                 output_dim=len(self.TAGS),
                                 word_embd_weights=self.word_embd_weights)
        self.model.summary()

        if os.path.exists(self.MODEL_PATH):
            print("loading model from: ", self.MODEL_PATH)
            self.model.load_weights(self.MODEL_PATH, by_name=True)
        else:
            print('Fitting...')
            self.model.fit_generator(
                generator=self.batch_generator(self.train_sentences, self.train_taggings, train_steps),
                steps_per_epoch=train_steps,
                epochs=self.EPOCHS,
                validation_data=self.batch_generator(self.valid_sentences, self.valid_taggings, valid_steps),
                validation_steps=valid_steps,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
                    keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=2),
                ],
                verbose=True,
            )

            self.model.save_weights(self.MODEL_PATH)

    # @classmethod
    def predict(self, data, *args, **kwargs):
        """
        This method is for predicting tags for tokens. The input is a file containing the samples
        The output is a list in which each element is a list of tags for each sentence.
        For example:
         [
        ['B-ORG','O','B-MISC','O','O','O','B-MISC','O','O'],
        ['B-PER','I-PER']
        ]
        :param data: the path of the file containing sentences
        :return: list of tags
        """
        test_sentences, test_taggings = self.load_data(data)
        test_steps = (len(test_sentences) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        # eps = 1e-6
        predicts = []
        true_tags = []
        # total_pred, total_true, matched_num = 0, 0, 0.0
        for inputs, batch_taggings in self.batch_generator(test_sentences, test_taggings, test_steps, training=False):
            predict = self.model.predict_on_batch(inputs)
            predict = numpy.argmax(predict, axis=2).tolist()
            for i, pred in enumerate(predict):
                predicts.append((list(map(lambda x: self.fromValueToKey(x), pred[:len(batch_taggings[i])]))))
                true_tags.append((list(map(lambda x: self.fromValueToKey(x), batch_taggings[i]))))
                pred = self.get_tags(pred[:len(batch_taggings[i])])
                true = self.get_tags(batch_taggings[i])
                self.total_pred += len(pred)
                self.total_true += len(true)
                self.matched_num += sum([1 for tag in pred if tag in true])
        return predicts, true_tags

    def fromValueToKey(self, value):
        for tag, index in self.TAGS.items():
            if index == value:
                return tag

    def batch_generator(self, sentences, taggings, steps, training=True):
        while True:
            for i in range(steps):
                batch_sentences = sentences[self.BATCH_SIZE * i:min(self.BATCH_SIZE * (i + 1), len(sentences))]
                batch_taggings = taggings[self.BATCH_SIZE * i:min(self.BATCH_SIZE * (i + 1), len(taggings))]
                word_input, char_input = get_batch_input(
                    batch_sentences,
                    self.max_word_len,
                    self.word_dict,
                    self.char_dict,
                    word_ignore_case=True,
                    char_ignore_case=False
                )
                if not training:
                    yield [word_input, char_input], batch_taggings
                    continue
                sentence_len = word_input.shape[1]
                for j in range(len(batch_taggings)):
                    batch_taggings[j] = batch_taggings[j] + [0] * (sentence_len - len(batch_taggings[j]))
                batch_taggings = self.to_categorical_tensor(numpy.asarray(batch_taggings), len(self.TAGS))
                yield [word_input, char_input], batch_taggings
            if not training:
                break

    def to_categorical_tensor(self, x3d, n_cls):
        batch_size, n_rows = x3d.shape
        x1d = x3d.ravel()
        y1d = keras.utils.to_categorical(x1d, num_classes=n_cls)
        y4d = y1d.reshape([batch_size, n_rows, n_cls])
        return y4d

    def get_tags(self, tags):
        filtered = []
        for i in range(len(tags)):
            if tags[i] == 0:
                continue
            if tags[i] % 2 == 1:
                filtered.append({
                    'begin': i,
                    'end': i,
                    'type': i,
                })
            elif i > 0 and tags[i - 1] == tags[i] - 1:
                filtered[-1]['end'] += 1
        return filtered
