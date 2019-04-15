import numpy as np
import os
import copy
from .. import ner
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sklearn
from keras.models import load_model
from nltk import word_tokenize
from validation import addCharInformationPrediction, padSentence
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from prepro import readfile,createBatches,createMatrices, createMatricesPrediction, iterate_minibatches,addCharInformation,padding, getCasing
from keras.utils import Progbar
from keras.initializers import RandomUniform

class ner_blstm_cnn(ner.Ner):

    def __init__(self, epoch=70):
        """
        Set parameters for class variables
        """
        # Number of epochs
        self.epochs = epoch

        # ::Hard coded char lookup ::
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|": #ÌöÛ’ò˙<>・【】●■□，▲（一）·の→￥：在＊à
            self.char2Idx[c] = len(self.char2Idx)
        # :: Hard coded case lookup ::
        self.case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                         'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(self.case2Idx), dtype='float32')

    def build_model(self):
        """
        Build the LSTM model for training

        Args:
            None
        Returns:
            None
        Raises:
            None
        """
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=self.wordEmbeddings.shape[0], output_dim=self.wordEmbeddings.shape[1],
                          weights=[self.wordEmbeddings], trainable=False)(words_input)
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=self.caseEmbeddings.shape[1], input_dim=self.caseEmbeddings.shape[0],
                           weights=[self.caseEmbeddings], trainable=False)(casing_input)
        character_input = Input(shape=(None, 52,), name='char_input')
        embed_char_out = TimeDistributed(
            Embedding(len(self.char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name='char_embedding')(character_input)
        dropout = Dropout(0.5)(embed_char_out)
        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(
            dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
        char = TimeDistributed(Flatten())(maxpool_out)
        char = Dropout(0.5)(char)
        output = concatenate([words, casing, char])
        output = Bidirectional(LSTM(250, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
        output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'))(output)
        self.model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
        self.model.summary()

    def load_embeddings(self):
        """
        Load word embeddings (default: glove.6B.100d.txt)

        Args:
            None.
        Returns:
            None.
        Raises:
            None
        """

        # :: Read in word embeddings ::
        self.word2Idx = {}
        wordEmbeddings = []

        fEmbeddings = open("embeddings/glove.6B.100d.txt", encoding="utf-8")

        for line in fEmbeddings:
            split = line.strip().split(" ")
            word = split[0]

            if len(self.word2Idx) == 0:  # Add padding+unknown
                self.word2Idx["PADDING_TOKEN"] = len(self.word2Idx)
                vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
                wordEmbeddings.append(vector)

                self.word2Idx["UNKNOWN_TOKEN"] = len(self.word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                wordEmbeddings.append(vector)

            if split[0].lower() in self.words:
                vector = np.array([float(num) for num in split[1:]])
                wordEmbeddings.append(vector)
                self.word2Idx[split[0]] = len(self.word2Idx)

        self.wordEmbeddings = np.array(wordEmbeddings)

        # Save embeddings for future use
        self.idx2Label = {v: k for k, v in self.label2Idx.items()}


    def load_models(self, modelpath=None):
        """
        Load custom model

        Args:
            modelpath (str): The path to the weights file
        Returns:
            None.
        Raises:
            None
        """
        if not modelpath:
            modelpath = os.path.join(os.path.expanduser('~'), '.ner_model')
        self.model = load_model(os.path.join(modelpath, "model.h5"))
        # loading word2Idx
        self.word2Idx = np.load(os.path.join(modelpath, "word2Idx.npy")).item()
        # loading idx2Label
        self.idx2Label = np.load(os.path.join(modelpath, "idx2Label.npy")).item()

    def createTensor(self, sentence, word2Idx, case2Idx, char2Idx):
        """
        Create a tensor to make run the model on and make predictions

        Args:
            sentence (list) : a list of lists consisting of each word and its corresponding characters
            word2Idx (dict) : {key: value}, dictionary with words as key mapped to the word id
            case2Idx (dict) : {key: value}, dictionary with case information as key mapped to the its correponding id
            char2Idx (dict) : {key: value}, dictionary with character as key along mapped to the character id
        Returns:
            tensor (list) : a list of lists consisting of indices for words, chars and case information.
        Raises:
            None
        """
        unknownIdx = word2Idx['UNKNOWN_TOKEN']

        wordIndices = []
        caseIndices = []
        charIndices = []

        for word, char in sentence:
            word = str(word)
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
            charIdx = []
            for x in char:
                if x in char2Idx.keys():
                    charIdx.append(char2Idx[x])
                else:
                    charIdx.append(char2Idx['UNKNOWN'])
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)

        return [wordIndices, caseIndices, charIndices]

    def predict_text(self, text):
        """
        Predicts on the given input text
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
        Sentence = words = word_tokenize(text)
        Sentence = addCharInformationPrediction(Sentence)
        Sentence = padSentence(self.createTensor(Sentence, self.word2Idx, self.case2Idx, self.char2Idx))
        tokens, casing, char = Sentence
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = self.model.predict([tokens, casing, char], verbose=False)[0]
        pred = pred.argmax(axis=-1)
        pred = [self.idx2Label[x].strip() for x in pred]

        start = [None]*len(pred)
        span = [len(word) for word in words]
        predictions = list(map(list, zip(start, span, words, pred)))
        predictions = [tuple(preds) for preds in predictions]
        return predictions


    def predict_dataset(self, dataset):
        """
        Predicts on the given dataset
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

        dataset = addCharInformation(dataset, self.char2Idx)
        batch = padding(createMatricesPrediction(dataset, self.word2Idx, self.label2Idx, self.case2Idx, self.char2Idx))

        tokens = []
        span = []
        predLabels = []

        b = Progbar(len(batch))
        for i, data in enumerate(batch):
            token, casing, char, labels, currentSentence = data
            token = np.asarray([token])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = self.model.predict([token, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes

            for sentence in currentSentence:
                tokens.append(sentence)
                span.append(len(sentence))

            preds = [self.idx2Label[element] for element in pred]
            predLabels.extend(preds)
            b.update(i)
        b.update(i + 1)

        start = [None] * len(tokens)
        predictions = list(map(list, zip(start, span, tokens, predLabels)))
        predictions = [tuple(pred) for pred in predictions]
        return predictions

    #@NER.convert_ground_truth
    def convert_ground_truth(self, data, *args, **kwargs):
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

        # return ground_truth

        tokens = []
        labels = []
        span = []

        for lines in data:
            for tuples in lines:
                tokens.append(tuples[0])
                labels.append(tuples[1])
                span.append(len(tuples[0]))

        start = [None] * len(tokens)
        ground_truth = list(map(list, zip(start, span, tokens, labels)))
        ground_truth = [tuple(ground) for ground in ground_truth]

        return ground_truth

    #@NER.overrides
    def read_dataset(self, fileNames, *args, **kwargs):
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

        trainSentences = readfile(fileNames['train'])
        devSentences = readfile(fileNames['valid'])
        testSentences = readfile(fileNames['test'])

        data = {'train':copy.deepcopy(trainSentences), 'valid':copy.deepcopy(devSentences), 'test':copy.deepcopy(testSentences)}

        labelSet = set()
        self.words = {}

        trainSentences = addCharInformation(trainSentences, self.char2Idx)
        devSentences = addCharInformation(devSentences, self.char2Idx)
        testSentences = addCharInformation(testSentences, self.char2Idx)

        for dataset in [trainSentences, devSentences, testSentences]:
            for sentence in dataset:
                for token, char, label in sentence:
                    labelSet.add(label)
                    self.words[token.lower()] = True

        # :: Create a mapping for the labels ::
        self.label2Idx = {}
        for label in labelSet:
            self.label2Idx[label] = len(self.label2Idx)

        return data

    #@NER.overrides
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

        # Load the word embeddings
        self.load_embeddings()

        # Prepare the model for training
        trainSentences = addCharInformation(copy.deepcopy(data['train']), self.char2Idx)
        train_set = padding(createMatrices(trainSentences, self.word2Idx, self.label2Idx, self.case2Idx, self.char2Idx))

        # Create mini-batches for training

        train_batch, train_batch_len = createBatches(train_set)

        # Build the model to train

        self.build_model()
        #self.model = load_model("./models/model.h5")

        ground = self.convert_ground_truth(data['valid'])

        # Train the model
        for epoch in range(self.epochs):
            print("Epoch %d/%d" % (epoch, self.epochs))
            a = Progbar(len(train_batch_len))
            for i, batch in enumerate(iterate_minibatches(train_batch, train_batch_len)):
                labels, tokens, casing, char = batch
                self.model.train_on_batch([tokens, casing, char], labels)
                a.update(i)
            predictions = self.predict_dataset(copy.deepcopy(data['valid']))
            P, R, F = self.evaluate(predictions, ground)
            print('Precision: %s, Recall: %s, F1: %s' % (P, R, F))
            a.update(i + 1)
            print(' ')

        self.model.save("models/model.h5")
        np.save("models/idx2Label.npy", self.idx2Label)
        np.save("models/word2Idx.npy", self.word2Idx)

    #@NER.overrides
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
            return self.predict_dataset(copy.deepcopy(data))

    #@NER.overrides
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

        predicted_labels = [predicted[3] for predicted in predictions]
        ground_labels = [true_labels[3] for true_labels in groundTruths]

        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_set = list(self.label2Idx.keys())
        label_encoder.fit(label_set)

        mapping = list(label_encoder.classes_)
        labels = []

        for i, label in enumerate(mapping):
            if label!='O':
                labels.append(i)

        ground_labels = label_encoder.transform(ground_labels)
        predicted_labels = label_encoder.transform(predicted_labels)

        precision = sklearn.metrics.precision_score(ground_labels, predicted_labels, average='micro', labels=labels)
        recall = sklearn.metrics.recall_score(ground_labels, predicted_labels, average='micro', labels=labels)
        f1 = sklearn.metrics.f1_score(ground_labels, predicted_labels, average='micro', labels=labels)
        return precision, recall, f1


"""
# Sample workflow:

inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

myModel = module_ner_lstm_cnn(DITKModel_NER)  # instatiate the class
data = myModel.read_dataset(inputFiles)  # read in a dataset for training

myModel.train(train_data)  # trains the model and stores model state in object properties or similar

predictions = myModel.predict(test_data)  # generate predictions! output format will be same for everyone

test_labels = myModel.convert_ground_truth(test_data)  <-- need ground truth labels need to be in same format as predictions!

P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1

print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
"""