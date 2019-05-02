import numpy as np
from src.datasetProperties import datasetProperties
from src.alphabet import CharBasedNERAlphabet

class CharBasedNERDataset:
    NULL_LABEL = '0'
    BASE_LABELS = [NULL_LABEL]


    def __init__(self,data):
        self.texts = self.get_texts(data['train'])
        # self.sentence_max_length_train = 516
        self.alphabet = CharBasedNERAlphabet(self.texts)
        self.labels = self.BASE_LABELS + self.get_labels(data['train'])
        self.num_labels = len(self.labels)
        self.num_to_label = {}
        self.label_to_num = {}

        self.init_mappings()

    def get_texts(self,train_data):
        """ Implement with own data source. """

        self.sentence_max_length_train = 0
        sentences = list()

        a =[]
        for l in train_data:
            if len(l):
                a.append(l)
            else:  # emtpy line
                if len(a):
                    ws = ""
                    for el in a:
                        ws +=el[0]+" "
                    if(len(ws)>self.sentence_max_length_train):
                        self.sentence_max_length_train = len(ws)
                    sentences.append(ws[0:len(ws)-1])

                a = []
        #for last line
        if len(a):
            ws = ""
            for el in a:
                ws += el[0] + " "
            if (len(ws) > self.sentence_max_length_train):
                self.sentence_max_length_train = len(ws)
            sentences.append(ws[0:len(ws) - 1])



        return sentences


    def get_x_y(self, dataset_name='all'):



        if dataset_name =="train":
            dataset_tuple = self.get_texts_and_labels(dataset_name)

        if dataset_name == "dev":
            dataset_tuple = self.get_texts_and_labels(dataset_name)


        if dataset_name == "test":
            dataset_tuple = self.get_texts_and_labels(dataset_name)


        return dataset_tuple




    def get_labels(self,train_data):


        labels = set()

        for line in train_data:
            if(len(line)):
                labels.add(line[3].lower())


        return list(labels)



    def str_to_x(self, s, maxlen):
        x = np.zeros(maxlen)
        for c, char in enumerate(s[:maxlen]):
            x[c] = self.alphabet.get_char_index(char)
        return x.reshape((-1, maxlen))

    def x_to_str(self, x):
        return [[self.alphabet.num_to_char[i] for i in row] for row in x]

    def y_to_labels(self, y):
        Y = []
        for row in y:
            Y.append([self.num_to_label[np.argmax(one_hot_labels)] for one_hot_labels in row])
        return Y

    def init_mappings(self):
        for num, label in enumerate(self.labels):
            self.num_to_label[num] = label
            self.label_to_num[label] = num


    def get_texts_and_labels(self,dataset_name):

        data = []
        size_of_data=0
        max_length = 0

        if(dataset_name == 'train'):
           data = datasetProperties.train_data
           size_of_data = datasetProperties.no_of_sentences_train
           max_length =datasetProperties.max_length_train

        if(dataset_name =='dev'):
            data = datasetProperties.dev_data
            size_of_data = datasetProperties.no_of_sentences_dev
            max_length = datasetProperties.max_length_dev

        if(dataset_name =='test'):
            data = datasetProperties.test_data
            size_of_data = datasetProperties.no_of_sentences_test
            max_length = datasetProperties.max_length_test




        tensor_x = np.full((size_of_data,max_length),0)
        tensor_y = np.full((size_of_data,max_length,self.num_labels),0)


        row = 0
        col =0


        sentence_data = []
        for line in data:
            if(len(line)):
                sentence_data.append(line)
            else:#sentence is over
                if(len(sentence_data)):
                    for word in sentence_data:
                        # if(word[3].lower()!= 'o'):
                        #     tag = word[3].split('-')
                        #     word_tag = "i-"+tag[1].lower()
                        # else:
                        word_tag = word[3].lower()


                        #make tensor_x
                        for c in word[0]:
                            tensor_x[row][col] = self.alphabet.get_char_index(c)
                            tensor_y[row][col] = self.getLabelTensor(c,word_tag)
                            col+=1

                        #for space after every word in sentence
                        tensor_x[row][col] = self.alphabet.get_char_index(' ')
                        col+=1


                row+=1
                col=0
                sentence_data = []


        result_tuple = (tensor_x,tensor_y)

        return result_tuple


    def getLabelTensor(self,c,word_tag):

        one_hot_encoding = [0] * self.num_labels

        one_hot_encoding[self.label_to_num[word_tag]] = 1

        return one_hot_encoding
