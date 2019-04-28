import os
import numpy as np
import re
import optparse
import itertools
from collections import OrderedDict
from src.utils import create_input
import src.loader as loader

from src.utils import evaluate, eval_script, eval_temp, save_mappings, reload_mappings
from src.loader import word_mapping, char_mapping, tag_mapping, pt_mapping
from src.loader import update_tag_scheme, prepare_dataset
from src.loader import augment_with_pretrained
from gensim.models import word2vec
from src.GRAMCNN import GRAMCNN

import tensorflow as tf

models_path = "./models"


class GramCnnNet:
    def __init__(self):
        
        self.train_path = "./dataset/CHEM/train.tsv"
        self.dev = "./dataset/CHEM/dev.tsv"
        self.test = "./dataset/CHEM/test.tsv"
        self.pre_emb = "./dataset/vectorfile/PubMed-shuffle-win-30.bin"

        self.dropout = 0.5
        self.word_lstm_dim = 675
        self.word_dim = 200
        self.hidden_layer = 7
        self.kernel_size = "2,3,4"
        self.kernel_num = "40,40,40"
        self.padding = 1
        self.pts = 0
        self.tag_scheme = "iob"

        self.lower = 0
        self.zeros = 0
        self.char_dim = 1
        self.char_lstm_dim = 1
        self.char_bidirect = 1
        self.word_bidirect = 1
        self.all_emb = 1
        self.cap_dim = 0
        self.crf = 1
        self.lr_method = "sgd-lr_.005"
        self.use_word = 1
        self.use_char = 1
        self.reloaded = 1
        
        # Parse parameters
        parameters = OrderedDict()
        #IOB OR IOEB
        parameters['padding'] = self.padding == 1
        parameters['tag_scheme'] = self.tag_scheme
        parameters['lower'] = self.lower == 1
        parameters['zeros'] = self.zeros == 1
        parameters['char_dim'] = self.char_dim
        parameters['char_lstm_dim'] = self.char_lstm_dim
        parameters['char_bidirect'] = self.char_bidirect == 1
        parameters['word_dim'] = self.word_dim
        parameters['word_lstm_dim'] = self.word_lstm_dim
        parameters['word_bidirect'] = self.word_bidirect == 1
        parameters['pre_emb'] = self.pre_emb
        parameters['all_emb'] = self.all_emb == 1
        parameters['cap_dim'] = self.cap_dim
        parameters['crf'] = self.crf == 1
        parameters['dropout'] = self.dropout
        parameters['lr_method'] = self.lr_method
        parameters['use_word'] = self.use_word == 1
        parameters['use_char'] = self.use_char == 1
        parameters['hidden_layer'] = self.hidden_layer
        parameters['reload'] = self.reloaded == 1
        parameters['kernels'] = [2,3,4,5] if type(self.kernel_size) == str else map(lambda x : int(x), self.kernel_size)
        parameters['num_kernels'] = [100,100,100,100] if type(self.kernel_num) == str else map(lambda x : int(x), self.kernel_num)
        parameters['pts'] = self.pts == 1
        
        self.parameters = parameters
        self.model_name = "chemner_2"
        
    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        pass

    def read_dataset(self, file_dict, dataset_name):  # <--- implemented PER class
        
        self.train_path = file_dict + dataset_name[0]
        self.dev = file_dict + dataset_name[1]
        self.test = file_dict + dataset_name[2]
        
        
        if 'bin' in self.parameters['pre_emb']:
            self.wordmodel = word2vec.Word2Vec.load_word2vec_format(self.parameters['pre_emb'], binary=True)
        else:
            self.wordmodel = word2vec.Word2Vec.load_word2vec_format(self.parameters['pre_emb'], binary=False)
            
        # Data parameters
        lower = self.parameters['lower']
        zeros = self.parameters['zeros']
        tag_scheme = self.parameters['tag_scheme']

        # Load sentences
        train_sentences = loader.load_sentences(self.train_path, self.lower, self.zeros)
        dev_sentences = loader.load_sentences(self.dev, self.lower, self.zeros)

        avg_len = sum([len(i) for i in train_sentences]) / float(len(train_sentences))
        print "train avg len: %d" % (avg_len)

        if os.path.isfile(self.test):
            test_sentences = loader.load_sentences(self.test, self.lower, self.zeros)
        
        '''
        # Sample
        train_sentences = train_sentences[:200]
        dev_sentences = dev_sentences[:200]
        test_sentences = test_sentences[:50]
        '''
        
        
        
        # Use selected tagging scheme (IOB / IOBES)
        update_tag_scheme(train_sentences, self.tag_scheme)
        update_tag_scheme(dev_sentences, self.tag_scheme)
        if os.path.isfile(self.test):
            update_tag_scheme(test_sentences, self.tag_scheme)

        dt_sentences = []
        if os.path.isfile(self.test):
            dt_sentences = dev_sentences + test_sentences
        else:
            dt_sentences = dev_sentences
            
        # Create a dictionary / mapping of words
        # If we use pretrained embeddings, we add them to the dictionary.
        self.word_to_id = []
        self.char_to_id = []
        self.pt_to_id = []
        self.tag_to_id = []
        
        if not self.parameters['reload']:
            if self.parameters['pre_emb']:
                # mapping of words frenquency decreasing
                self.dico_words_train = word_mapping(train_sentences, self.lower)[0]
                self.dico_words, self.word_to_id, self.id_to_word = augment_with_pretrained(
                    self.dico_words_train.copy(),
                    self.wordmodel,
                    list(itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in dt_sentences])
                    ) if not self.parameters['all_emb'] else None
                )
            else:
                self.dico_words, self.word_to_id, self.id_to_word = word_mapping(train_sentences, self.lower)
                self.dico_words_train = dico_words


            # Create a dictionary and a mapping for words / POS tags / tags
            self.dico_chars, self.char_to_id, self.id_to_char = char_mapping(train_sentences)
            self.dico_tags, self.tag_to_id, self.id_to_tag = tag_mapping(train_sentences)
            self.dico_pts, self.pt_to_id, self.id_to_pt = pt_mapping(train_sentences + dev_sentences)
            if not os.path.exists(os.path.join(models_path, self.model_name)):
                    os.makedirs(os.path.join(models_path,self.model_name))
            save_mappings(os.path.join(models_path, self.model_name, 'mappings.pkl'), self.word_to_id, self.char_to_id, self.tag_to_id, self.pt_to_id, self.dico_words, self.id_to_tag)
        else:
            self.word_to_id, self.char_to_id, self.tag_to_id, self.pt_to_id, self.dico_words, self.id_to_tag = reload_mappings(os.path.join(models_path, self.model_name, 'mappings.pkl'))
            self.dico_words_train = self.dico_words
            self.id_to_word = {v: k for k, v in self.word_to_id.items()}
            
        # Index data
        m3 = 0
        train_data,m1 = prepare_dataset(
            train_sentences, self.word_to_id, self.char_to_id, self.tag_to_id, self.pt_to_id, self.lower
        )
        dev_data,m2 = prepare_dataset(
            dev_sentences, self.word_to_id, self.char_to_id, self.tag_to_id, self.pt_to_id, self.lower
        )
        if os.path.isfile(self.test):
            test_data,m3 = prepare_dataset(
                test_sentences, self.word_to_id, self.char_to_id, self.tag_to_id, self.pt_to_id, self.lower
            )

        self.max_seq_len = max(m1,m2,m3)
        print "max length is %i" % (self.max_seq_len)

        print "%i / %i  sentences in train / dev." % (
           len(train_data), len(dev_data))
        
        return (train_data, dev_data, test_data, test_sentences)
            
            
    def train(self, data):  # <--- implemented PER class
        #
        # Train network
        #
        train_data, dev_data, test_data, test_sentences = data
        
        singletons = set([self.word_to_id[k] for k, v
                          in self.dico_words_train.items() if v == 1])

        n_epochs = 100  # number of epochs over the training set
        freq_eval = 2000  # evaluate on dev every freq_eval steps
        best_dev = -np.inf
        best_test = -np.inf
        count = 0
        
        #initilaze the embedding matrix
        word_emb_weight = np.zeros((len(self.dico_words), self.parameters['word_dim']))
        c_found = 0
        c_lower = 0
        c_zeros = 0
        n_words = len(self.dico_words)
        for i in xrange(n_words):
                word = self.id_to_word[i]
                if word in self.wordmodel:
                    word_emb_weight[i] = self.wordmodel[word]
                    c_found += 1
                elif re.sub('\d', '0', word) in self.wordmodel:
                    word_emb_weight[i] = self.wordmodel[
                        re.sub('\d', '0', word)
                    ]
                    c_zeros += 1

        print 'Loaded %i pretrained embeddings.' % len(self.wordmodel.vocab)
        print ('%i / %i (%.4f%%) words have been initialized with '
               'pretrained embeddings.') % (
                    c_found + c_lower + c_zeros, n_words,
                    100. * (c_found + c_lower + c_zeros) / n_words
              )
        print ('%i found directly, %i after lowercasing, '
               '%i after lowercasing + zero.') % (
                  c_found, c_lower, c_zeros
              )
        
        self.gramcnn = GRAMCNN(n_words, len(self.char_to_id), len(self.pt_to_id),
                    use_word = self.parameters['use_word'],
                    use_char = self.parameters['use_char'],
                    use_pts = self.parameters['pts'],
                    num_classes = len(self.id_to_tag),
                    word_emb = self.parameters['word_dim'],
                    drop_out = self.parameters['dropout'],
                    word2vec = word_emb_weight, feature_maps=self.parameters['num_kernels'],#,200,200, 200,200],
                    kernels = self.parameters['kernels'], hidden_size = self.parameters['word_lstm_dim'], hidden_layers = self.parameters['hidden_layer'],
                    padding = self.parameters['padding'], max_seq_len = self.max_seq_len, train_size = len(train_data))

        for epoch in xrange(n_epochs):
            epoch_costs = []
            print "Starting epoch %i..." % epoch
            for i, index in enumerate(np.random.permutation(len(train_data))):
                inputs, word_len = create_input(train_data[index], self.parameters, True, singletons,
                    padding = self.parameters['padding'], max_seq_len = self.max_seq_len, use_pts = self.parameters['pts'] )

                assert inputs['char_for']
                assert inputs['word']
                assert inputs['label']

                # break
                if len(inputs['label']) == 1:
                    continue
                train_loss = []
                temp = []
                temp.append(word_len)
                batch_loss = self.gramcnn.train(inputs, temp)

                train_loss.append(batch_loss)
                
                if(i % 10 == 0 and i != 0):
                    print( "Epoch[%d], "%(epoch) + "Iter " + str(i))
                    
                if(i % 500 == 0 and i != 0):
                    print( "Epoch[%d], "%(epoch) + "Iter " + str(i) +                             ", Minibatch Loss= " + "{:.6f}".format(np.mean(train_loss[-500:])))
                    train_loss = []
                
                '''
                if i % 2000 == 0 and i != 0:
                    dev_score = evaluate(parameters, gramcnn, dev_sentences,
                                         dev_data, id_to_tag, padding = parameters['padding'],
                                         max_seq_len = max_seq_len, use_pts = parameters['pts'])
                    print "dev_score_end"
                    print "Score on dev: %.5f" % dev_score
                    if dev_score > best_dev:
                        best_dev = dev_score
                        print "New best score on dev."
                        print "Saving model to disk..."
                        gramcnn.save(models_path ,self.model_name)
                    if os.path.isfile(opts.test):
                        if i % 8000 == 0 and i != 0:
                            test_score = evaluate(parameters, gramcnn, test_sentences,
                                                  test_data, id_to_tag, padding = parameters['padding'],
                                                  max_seq_len = max_seq_len, use_pts = parameters['pts'])
                            print "Score on test: %.5f" % test_score
                            if test_score > best_test:
                                best_test = test_score
                                print "New best score on test."
                '''
        

    def predict(self, data, *args, **kwargs):  # <--- implemented PER class WITH requirement on OUTPUT format!
        train_data, dev_data, test_data, test_sentences = data
        
        singletons = set([self.word_to_id[k] for k, v
                          in self.dico_words_train.items() if v == 1])

        n_epochs = 1000  # number of epochs over the training set
        freq_eval = 2000  # evaluate on dev every freq_eval steps
        best_dev = -np.inf
        best_test = -np.inf
        count = 0
        
        #initilaze the embedding matrix
        word_emb_weight = np.zeros((len(self.dico_words), self.parameters['word_dim']))
        c_found = 0
        c_lower = 0
        c_zeros = 0
        n_words = len(self.dico_words)
        for i in xrange(n_words):
                word = self.id_to_word[i]
                if word in self.wordmodel:
                    word_emb_weight[i] = self.wordmodel[word]
                    c_found += 1
                elif re.sub('\d', '0', word) in self.wordmodel:
                    word_emb_weight[i] = self.wordmodel[
                        re.sub('\d', '0', word)
                    ]
                    c_zeros += 1

        print 'Loaded %i pretrained embeddings.' % len(self.wordmodel.vocab)
        print ('%i / %i (%.4f%%) words have been initialized with '
               'pretrained embeddings.') % (
                    c_found + c_lower + c_zeros, n_words,
                    100. * (c_found + c_lower + c_zeros) / n_words
              )
        print ('%i found directly, %i after lowercasing, '
               '%i after lowercasing + zero.') % (
                  c_found, c_lower, c_zeros
              )
        
        test_score, output_path = evaluate(self.parameters, self.gramcnn, test_sentences, test_data, self.id_to_tag, remove = False, max_seq_len = self.max_seq_len, padding = self.parameters['padding'], use_pts = self.parameters['pts'])
        
        return output_path

    def evaluate(self, predictions, groundTruths, *args,
                 **kwargs):  # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
        
        train_data, dev_data, test_data, test_sentences = groundTruths
        
        singletons = set([self.word_to_id[k] for k, v
                          in self.dico_words_train.items() if v == 1])

        n_epochs = 1000  # number of epochs over the training set
        freq_eval = 2000  # evaluate on dev every freq_eval steps
        best_dev = -np.inf
        best_test = -np.inf
        count = 0
        
        #initilaze the embedding matrix
        word_emb_weight = np.zeros((len(self.dico_words), self.parameters['word_dim']))
        c_found = 0
        c_lower = 0
        c_zeros = 0
        n_words = len(self.dico_words)
        for i in xrange(n_words):
                word = self.id_to_word[i]
                if word in self.wordmodel:
                    word_emb_weight[i] = self.wordmodel[word]
                    c_found += 1
                elif re.sub('\d', '0', word) in self.wordmodel:
                    word_emb_weight[i] = self.wordmodel[
                        re.sub('\d', '0', word)
                    ]
                    c_zeros += 1

        print 'Loaded %i pretrained embeddings.' % len(self.wordmodel.vocab)
        print ('%i / %i (%.4f%%) words have been initialized with '
               'pretrained embeddings.') % (
                    c_found + c_lower + c_zeros, n_words,
                    100. * (c_found + c_lower + c_zeros) / n_words
              )
        print ('%i found directly, %i after lowercasing, '
               '%i after lowercasing + zero.') % (
                  c_found, c_lower, c_zeros
              )
        
        test_score, output_path = evaluate(self.parameters, self.gramcnn, test_sentences, test_data, self.id_to_tag, remove = False, max_seq_len = self.max_seq_len, padding = self.parameters['padding'], use_pts = self.parameters['pts'])
        
        return test_score


    def save_model(self, filepath):
        self.gramcnn.save(models_path ,self.model_name)

    

    def load_model(self, filepath):
        
        n_epochs = 100  # number of epochs over the training set
        freq_eval = 2000  # evaluate on dev every freq_eval steps
        best_dev = -np.inf
        best_test = -np.inf
        count = 0
        #self.max_seq_len = m3 if m3 > 200 else 200

        #initilaze the embedding matrix
        word_emb_weight = np.zeros((len(self.dico_words), self.parameters['word_dim']))
        n_words = len(self.dico_words)



        self.gramcnn = GRAMCNN(n_words, len(self.char_to_id), len(self.pt_to_id),
                            use_word = self.parameters['use_word'],
                            use_char = self.parameters['use_char'],
                            use_pts = self.parameters['pts'],
                            num_classes = len(self.tag_to_id),
                            word_emb = self.parameters['word_dim'],
                            drop_out = 0,
                            word2vec = word_emb_weight,feature_maps=self.parameters['num_kernels'],#,200,200, 200,200],
                            kernels=self.parameters['kernels'], hidden_size = self.parameters['word_lstm_dim'], hidden_layers = self.parameters['hidden_layer'],
                            padding = self.parameters['padding'], max_seq_len = self.max_seq_len)

        
        self.gramcnn.load(models_path ,self.model_name)
        
        
        


def main(input_file):
    GramCnnNet_instance = GramCnnNet()
        
    file_dict = "./dataset/CHEMDNER/"
    dataset_name = ["train.tsv", "dev.tsv", "test.tsv"]
    read_data = GramCnnNet_instance.read_dataset(file_dict, dataset_name)

    GramCnnNet_instance.train(read_data)
    
    output_file = GramCnnNet_instance.predict(read_data)
    print "Output file has been created at: {}".format(output_file)
    
    f1_score = GramCnnNet_instance.evaluate(None, read_data)
    print("f1: {}".format(f1_score))

    return output_file

if __name__ == '__main__':
    input_file = "./dataset/CHEMDNER/test.tsv"
    main(input_file)

"""
# Sample workflow:
file_dict = {
                "train": {"data" : "/home/sample_train.txt"},
                "dev": {"data" : "/home/sample_dev.txt"},
                "test": {"data" : "/home/sample_test.txt"},
             }
dataset_name = 'CONLL2003'
# instatiate the class
myModel = myClass() 
# read in a dataset for training
data = myModel.read_dataset(file_dict, dataset_name)  
myModel.train(data)  # trains the model and stores model state in object properties or similar
predictions = myModel.predict(data['test'])  # generate predictions! output format will be same for everyone
test_labels = myModel.convert_ground_truth(data['test'])  <-- need ground truth labels need to be in same format as predictions!
P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1
print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
"""
