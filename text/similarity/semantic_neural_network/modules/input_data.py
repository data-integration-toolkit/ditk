# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from modules.configs import *
from modules.prepare_text import prepare_text


class InputData:
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y


class ProcessInputData:
    def __init__(self, tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        self.max_sentence_length = -1

    def pre_process_data(self, df):
        sentences_1 = []
        sentences_2 = []
        labels = []
        for index, row in df.iterrows():
            sentences_1.append(prepare_text(row['s1']))
            sentences_2.append(prepare_text(row['s2']))
            label = float(row['label'])
            labels.append(round(label, 1))

        return sentences_1, sentences_2, labels

    def get_samples(self, sentences_1, sentences_2, label, rescaling_output=True):
        # Prepare the neural network inputs
        input_sentences_1 = self.tokenizer.texts_to_sequences(sentences_1)
        input_sentences_2 = self.tokenizer.texts_to_sequences(sentences_2)
        x1 = pad_sequences(input_sentences_1, self.max_sentence_length)
        x2 = pad_sequences(input_sentences_2, self.max_sentence_length)
        y = np.array(label)
        if rescaling_output:
            y = np.clip(y, 1, 5)  # paper definition
            y = (y - 1) / 4  # WARNING: LABEL RESCALING
            print(y)
        return InputData(x1, x2, y)

    def get_input_from_pair(self, sentence_1, sentence_2, sentence_length):
        # Prepare the neural network inputs
        input_sentences_1 = self.tokenizer.texts_to_sequences([prepare_text(sentence_1)])
        input_sentences_2 = self.tokenizer.texts_to_sequences([prepare_text(sentence_2)])
        x1 = pad_sequences(input_sentences_1, sentence_length)
        x2 = pad_sequences(input_sentences_2, sentence_length)
        return x1, x2

    def get_input_from_collection(self, sentences_1, sentences_2, sentence_length):
        preprocessed_sentences_1 = []
        preprocessed_sentences_2 = []
        for index in range(0, len(sentences_1)):
            preprocessed_sentences_1.append(prepare_text(sentences_1[index]))
            preprocessed_sentences_2.append(prepare_text(sentences_2[index]))

        # Prepare the neural network inputs
        input_sentences_1 = self.tokenizer.texts_to_sequences(preprocessed_sentences_1)
        input_sentences_2 = self.tokenizer.texts_to_sequences(preprocessed_sentences_2)
        x1 = pad_sequences(input_sentences_1, sentence_length)
        x2 = pad_sequences(input_sentences_2, sentence_length)
        return x1, x2

    def prepare_data(self, data_frames, dataset_name):
        labels = []
        sentences_1 = []
        sentences_2 = []
        for data_frame in data_frames:
            train_sentences_1, train_sentences_2, train_labels = self.pre_process_data(data_frame)
            self.tokenizer.fit_on_texts(train_sentences_1)
            self.tokenizer.fit_on_texts(train_sentences_2)
            sentences_1.append(train_sentences_1)
            sentences_2.append(train_sentences_2)

            labels.append(train_labels)

        self.word_index = self.tokenizer.word_index
        self.vocabulary_size = len(self.word_index)
        self.save_tokenizer(dataset_name)

        max_sentence_length = 0
        # The size of the input sequence is the size of the largest sequence of the input dataset
        for sentence_vec_side in [sentences_1, sentences_2]:
            for sentence_vec in sentence_vec_side:
                for sentence in sentence_vec:
                    sentence_length = len(sentence.split())
                    if sentence_length > max_sentence_length:
                        max_sentence_length = sentence_length


        self.max_sentence_length = 100

        results = []
        for i in range(0, len(data_frames)):
            input_data = self.get_samples(sentences_1[i], sentences_2[i], labels[i])
            results.append(input_data)
        return results

    def save_tokenizer(self, dataset):
        tokenizer_file = "tokenizer_%s.pickle" % (dataset)
        tokenizer_filepath = os.path.join(BASE_PATH+'tokenizers', tokenizer_file)
        if not os.path.exists(tokenizer_filepath):
            with open(tokenizer_filepath, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
