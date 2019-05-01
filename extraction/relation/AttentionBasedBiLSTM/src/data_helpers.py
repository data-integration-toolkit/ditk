import numpy as np
import pandas as pd
import nltk
import re

from src import utils


# from configure import FLAGS


def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def load_data_and_labels(unprocessed_data,dataType):
    data = []
    lines = unprocessed_data
    max_sentence_length = 0
    for idx in range(0, len(lines)):
        id = lines[idx][0]
        relation = lines[idx][-1]

        sentence = lines[idx][1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = nltk.word_tokenize(sentence)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        sentence = " ".join(tokens)

        data.append([id, sentence, relation])


    # print("max sentence length = {}\n".format(max_sentence_length))

    class2labels = dict()
    labels2class = dict()

    #TODO make utils class2label and label2class

    if(dataType =='train'):
        unique_labels = []
        for row in data:
            if row[2] not in unique_labels:
                unique_labels.append(row[2])


        labels2classfile = open("../res/label2class.txt","w+")
        class2labelsfile = open("../res/class2label.txt","w+")


        for number in range(0,len(unique_labels)):
            labels2class[number] = unique_labels[number]
            class2labels[unique_labels[number]] = number


            labels2classfile.write(str(number)+"\t"+unique_labels[number]+"\n")
            class2labelsfile.write(str(unique_labels[number])+"\t"+str(number)+"\n")


        labels2classfile.close()
        class2labelsfile.close()

        utils.class2label = class2labels
        utils.label2class = labels2class


    if(dataType =='test' or dataType == 'predict'):

        with open("../res/class2label.txt","r") as file:
            for line in file:
                (key,value) = line.split('\t')
                class2labels[key] = int(value.rstrip())
                labels2class[int(value.rstrip())] = key

        utils.class2label = class2labels
        utils.label2class = labels2class



    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [utils.class2label[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()


    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


