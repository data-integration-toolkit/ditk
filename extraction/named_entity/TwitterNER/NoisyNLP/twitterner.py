import ner
import os
import pickle
import six
import sys
from features import *
from utils import *
from models import *
from experiments import *
from run_ner import TwitterNER
from twokenize import tokenizeRawTweetText
import pandas as pd
import csv
from utils_refactored import *

main_dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
pkl_model_path = './tweet_model.pkl'
def load(filename, sep="\t", notypes=False):
    tag_count = defaultdict(int)
    sequences = []
    with open(filename) as fp:
        seq = []
        for line in fp:
          #print seq
          #print sequences
          line = line.strip()
          if line:
            line = line.split(sep)
            if notypes:
              line[1] = line[1][0]
            try:
              tag_count[line[1]] += 1
              #print line
              seq.append(Tag(*line))
            except:
              pass
            else:
              sequences.append(seq)
              seq = []

        if seq:
            sequences.append(seq)
    return sequences, tag_count

def write_sequences(sequences, filename, sep="\t", to_bieou=True):
    with open(filename, "wb+") as fp:
        for seq in sequences:
            if to_bieou:
                seq = to_BIEOU(seq)
            for tag in seq:
                print >> fp, sep.join(tag).encode('utf-8')
            print >> fp, ""

def phrase_to_BIEOU(phrase):
    l = len(phrase)
    new_phrase = []
    for j, t in enumerate(phrase):
        new_tag = t.tag
        if l == 1:
            new_tag = "U%s" % t.tag[1:]
        elif j == l-1:
            new_tag = "E%s" % t.tag[1:]
        new_phrase.append(Tag(t.token, new_tag))
    return new_phrase

def to_BIEOU(seq, verbose=False):
    # TAGS B I E U O
    phrase = []
    new_seq = []
    for i, tag in enumerate(seq):
        if not phrase and tag.tag[0] == "B":
            phrase.append(tag)
            continue
        if tag.tag[0] == "I":
            phrase.append(tag)
            continue
        if phrase:
            if verbose:
                print "Editing phrase", phrase
            new_phrase = phrase_to_BIEOU(phrase)
            new_seq.extend(new_phrase)
            phrase = []
        new_seq.append(tag)
    if phrase:
        if verbose:
            print "Editing phrase", phrase
            new_phrase = phrase_to_BIEOU(phrase)
            new_seq.extend(new_phrase)
            phrase = []
        new_seq.append(tag)
    if phrase:
        if verbose:
            print "Editing phrase", phrase
        new_phrase = phrase_to_BIEOU(phrase)
        new_seq.extend(new_phrase)
        phrase = []
    return new_seq

def write_to_csv(input,filePath):
    fp=open(filePath,'w+')
    for each_input in input:
        text=""
        try:
            text+=each_input[0]+'\t'+each_input[1]+'\t'+each_input[2]+'\t'+each_input[3]+'\n'
            fp.write(text)
        except:
            pass

    fp.close()

class TwitterNER(ner.NER):
    def convert_ground_truth():
        pass
    def read_dataset(self, outdir, file_dict, vocab_file=None, sep="\t",notypes=None,dataset=None):
        project_path = main_dir_path + "/data/refactored/"
        if not os.path.isdir(project_path):
            os.makedirs(project_path)
            #print("Directory %s created." % outdir, file=sys.stderr)
        dev_files = file_dict['dev']
        train_files = file_dict['train']
        test_files = file_dict['test']
        if dataset == 'WNUT':
            self.dev_files = dev_files[0]
            self.train_files = train_files[0]
            self.test_files = test_files[0]
        elif dataset == 'ConLL':
            converted_train_dir = os.path.join(project_path,"converted_train.tsv")
            converted_test_dir = os.path.join(project_path,"converted_test.tsv")
            converted_dev_dir = os.path.join(project_path,"converted_dev.tsv")
            cleaned_train_dir = os.path.join(project_path,"cleaned_train.tsv")
            cleaned_test_dir = os.path.join(project_path,"cleaned_test.tsv")
            cleaned_dev_dir = os.path.join(project_path,"cleaned_dev.tsv")
            print (converted_train_dir)

            print train_files
            df = pd.read_csv(train_files[0]) #train_files given as csv, if more than one file in list, read entire list
            print (df)
            df.columns = ['A','B','C','D'] #Add arbitrary columns
            df.to_csv(converted_train_dir,header=False,index=False, sep='\t')
            #print (load_sequences(converted_train_dir))
            sequences, tag_count = load(converted_train_dir)
            write_sequences(sequences, cleaned_train_dir, to_bieou=True)

            df = pd.read_csv(test_files[0]) #test_files given as csv
            df.columns = ['A','B','C','D'] #Add arbitrary columns
            df.to_csv(converted_test_dir,header=False,index=False, sep='\t')
            sequences, tag_count = load(converted_test_dir, sep="\t")
            write_sequences(sequences, cleaned_test_dir, to_bieou=True)
            
            df = pd.read_csv(dev_files[0], error_bad_lines=False) #dev_files given as csv
            #print (df)
            df.columns = ['A','B','C','D'] #Add arbitrary columns
            df.to_csv(converted_dev_dir,header=False,index=False, sep='\t')
            sequences, tag_count = load(converted_dev_dir, sep="\t")
            write_sequences(sequences, cleaned_dev_dir, to_bieou=True)
            
            self.dev_files = cleaned_dev_dir
            self.train_files = cleaned_train_dir
            self.test_files = cleaned_test_dir

        elif dataset == 'OntoNotes':
            converted_train_dir = os.path.join(project_path,"converted_train_onto.tsv")
            converted_test_dir = os.path.join(project_path,"converted_test_onto.tsv")
            converted_dev_dir = os.path.join(project_path,"converted_dev_onto.tsv")
            cleaned_train_dir = os.path.join(project_path,"cleaned_train_onto.tsv")
            cleaned_test_dir = os.path.join(project_path,"cleaned_test_onto.tsv")
            cleaned_dev_dir = os.path.join(project_path,"cleaned_dev_onto.tsv")
            
            lines = [] 
            with open(train_files[0]) as fp:
                while True:
                    text=fp.readline()
                    if text=="":
                        break
                    lines.append(text)
            converted = conll2012_to_ditk(lines)
            print len(converted)
            write_to_csv(converted,converted_train_dir)
            df = pd.read_csv(converted_train_dir,sep='\t') #train_files given as csv
            print (df)
            df.columns = ['A','B','C','D'] #Add arbitrary columns
            df.to_csv(converted_train_dir,columns = ['A','D'],header=False,index=False, sep='\t')
            sequences, tag_count = load(converted_train_dir, sep="\t")
            #print (sequences)
            write_sequences(sequences, cleaned_train_dir, to_bieou=True)
           
            lines = []
            with open(test_files[0]) as fp:
                while True:
                    text=fp.readline()
                    if text=="":
                        break
                    lines.append(text)
            converted = conll2012_to_ditk(lines)
            write_to_csv(converted,converted_test_dir)
            df = pd.read_csv(converted_test_dir,sep='\t',engine='python',error_bad_lines=False)#test_files given as csv
            df.columns = ['A','B','C','D'] #Add arbitrary columns
            df.to_csv(converted_test_dir,columns = ['A','D'],header=False,index=False, sep='\t')
            sequences, tag_count = load(converted_test_dir, sep="\t")
            write_sequences(sequences, cleaned_test_dir, to_bieou=True)
         
            lines = []
            with open(dev_files[0]) as fp:
                while True:
                    text=fp.readline()
                    if text=="":
                        break
                    lines.append(text)
            converted = conll2012_to_ditk(lines)
            write_to_csv(converted,converted_dev_dir)
            df = pd.read_csv(converted_dev_dir,sep='\t') #test_files given as csv
            df.columns = ['A','B','C','D'] #Add arbitrary columns
            df.to_csv(converted_dev_dir,columns = ['A','D'],header=False,index=False, sep='\t')
            sequences, tag_count = load(converted_dev_dir, sep="\t")
            write_sequences(sequences, cleaned_dev_dir, to_bieou=True)

            self.dev_files = cleaned_dev_dir
            self.train_files = cleaned_train_dir
            self.test_files = cleaned_test_dir


        if not os.path.isdir(outdir):
            #print("Directory %s doesn't exist." % outdir, file=sys.stderr)
            os.makedirs(outdir)
            #print("Directory %s created." % outdir, file=sys.stderr)
        self.outdir = outdir
        self.notypes = notypes
        '''self.dev_files = dev_files
        self.train_files = train_files
        self.test_files = test_files'''
        train_list = []
        dev_list = []
        test_list = []
        train_list.append(self.train_files)
        dev_list.append(self.dev_files)
        test_list.append(self.test_files)

        print (train_list)
        self.train_files = train_list
        self.dev_files = dev_list
        self.test_files = test_list
        

        self.train_sequences = file2sequences(train_list, sep=sep ,notypes=notypes)
        if dev_files is not None:
            self.dev_sequences = file2sequences(dev_list, sep=sep ,notypes=notypes)
        if test_files is not None:
            self.test_sequences = file2sequences(test_list, sep=sep ,notypes=notypes)
        self.vocab = load_vocab(vocab_file)
        all_sequences = [[t[0] for t in seq] 
                        for seq in (self.train_sequences + self.dev_sequences + self.test_sequences)]
        #all_sequences = load_sequences(self.dev_files)
        #for (train_file, encoding) in self.train_files:
            #all_sequences.extend(load_sequences(train_file, sep="\t", encoding=encoding))
        self.all_tokens = [[t[0] for t in seq] for seq in all_sequences]
        print (len(self.train_sequences))
        print (len(self.dev_sequences))
        print (len(self.test_sequences))
        print ('Read dataset done.')
    
    def load_features(self):
        '''sep = '\t'
        notypes = 'None'
        self.train_sequences = file2sequences(self.train_files, sep=sep ,notypes=notypes)
        if dev_files is not None:
            self.dev_sequences = file2sequences(self.dev_files, sep=sep ,notypes=notypes)
        if test_files is not None:
            self.test_sequences = file2sequences(self.test_files, sep=sep ,notypes=notypes)
        self.vocab = load_vocab(vocab_file)
        all_sequences = [[t[0] for t in seq]
                        for seq in (self.train_sequences + self.dev_sequences + self.test_sequences)]
        #all_sequences = load_sequences(self.dev_files)
        #for (train_file, encoding) in self.train_files:
            #all_sequences.extend(load_sequences(train_file, sep="\t", encoding=encoding))
        self.all_tokens = [[t[0] for t in seq] for seq in all_sequences]
        print (len(self.train_sequences))
        print (len(self.dev_sequences))
        print (len(self.test_sequences))'''
        self.dict_features = DictionaryFeatures(dictionary_dir)

        if not os.path.exists(wordvec_file_processed):
            process_glovevectors(wordvec_file)
        self.wv_model = WordVectors(self.all_tokens, wordvec_file_processed)

        gimple_brown_cf = ClusterFeatures(gimple_twitter_brown_clusters_dir, cluster_type="brown")
        gimple_brown_cf.set_cluster_file_path(gimple_twitter_brown_clusters_dir)
        self.gimple_brown_clusters = gimple_brown_cf.read_clusters()

        test_enriched_data_brown_cluster_dir = enriched_brown_cluster_dir
        test_enriched_data_brown_cf = ClusterFeatures(test_enriched_data_brown_cluster_dir,
                                                      cluster_type="brown", n_clusters=100)
        test_enriched_data_brown_cf.set_cluster_file_path()
        self.test_enriched_data_brown_clusters = test_enriched_data_brown_cf.read_clusters()

        test_enriched_data_clark_cluster_dir = enriched_clark_cluster_dir
        test_enriched_data_clark_cf = ClusterFeatures(test_enriched_data_clark_cluster_dir,
                                                      cluster_type="clark", n_clusters=32)
        test_enriched_data_clark_cf.set_cluster_file_path()
        self.test_enriched_data_clark_clusters = test_enriched_data_clark_cf.read_clusters()

    def train(self):
        self.load_features()
        self.gen_model_data(proc_func=self.get_X_y_exp)
        self.fit_evaluate()
        self.save_model(self.model)

    def save_model(self,model):
        with open(pkl_model_path, "wb+") as fp:
            pickle.dump(model, fp)
    
    def load_model(self,filePath):
        with open((filePath), "rb") as pickle_file:
            self.model = pickle.load(pickle_file)
        return self.model

    def get_features(self, tokens):
        return sent2features(tokens, WORD_IDX=None, vocab=None,
                             dict_features=self.dict_features, vocab_presence_only=False,
                             window=4, interactions=True, dict_interactions=True,
                             lowercase=False, dropout=0,
                             word2vec_model=self.wv_model.model,
                             cluster_vocabs=[
                               self.gimple_brown_clusters,
                               self.test_enriched_data_brown_clusters,
                               self.test_enriched_data_clark_clusters
                             ])

    def get_X_y_exp(self, sequences):
        print sequences
        X = [sent2features(s, vocab=None,
                         dict_features=self.dict_features, vocab_presence_only=False,
                         window=4, interactions=True, dict_interactions=True,
                         lowercase=False, dropout=0, word2vec_model=self.wv_model.model,
                        cluster_vocabs=[
            self.gimple_brown_clusters,
            self.test_enriched_data_brown_clusters,
            self.test_enriched_data_clark_clusters
        ])
         for s in sequences]
        y = [sent2labels(s) for s in sequences]
        return X, y
   
    def get_X_y(self,sequences, proc_func, label="Data"):
        start = time.time()
        X, y = proc_func(sequences)
        end = time.time()
        process_time = end - start
        print("%s feature generation took: %s" % (label, datetime.timedelta(seconds=process_time)))
        return X, y

    def gen_model_data(self, proc_func):
        self.X_train, self.y_train = get_X_y(
            self.train_sequences, proc_func=proc_func, label="Train", **kwargs)
        self.X_dev, self.y_dev = get_X_y(
            self.dev_sequences, proc_func=proc_func, label="Dev", **kwargs)
        self.X_test, self.y_test = get_X_y(
            self.test_sequences, proc_func=proc_func, label="Test", **kwargs)
        print("Train: %s, %s\nDev: %s, %s\nTest: %s, %s" % (len(self.X_train), len(self.y_train),
                                                            len(self.X_dev), len(self.y_dev),
                                                            len(self.X_test), len(self.y_test)))
   
    def get_X_y(self,sequences, proc_func, label="Data"):
        start = time.time()
        X, y = proc_func(sequences)
        end = time.time()
        process_time = end - start
        print("%s feature generation took: %s" % (label, datetime.timedelta(seconds=process_time)))
        return X, y

    def gen_model_data(self, proc_func):
        self.X_train, self.y_train = get_X_y(
            self.train_sequences, proc_func=proc_func, label="Train")
        self.X_dev, self.y_dev = get_X_y(
            self.dev_sequences, proc_func=proc_func, label="Dev")
        self.X_test, self.y_test = get_X_y(
            self.test_sequences, proc_func=proc_func, label="Test")
        print("Train: %s, %s\nDev: %s, %s\nTest: %s, %s" % (len(self.X_train), len(self.y_train),
                                                            len(self.X_dev), len(self.y_dev),
                                                            len(self.X_test), len(self.y_test)))
    def fit_evaluate(self, notypes=None):
        #notypes = self.notypes
        #print self.notypes
        if notypes is None:
            notypes = self.notypes
        #print self.notypes
        self.model = CRFModel()
        self.model.fit(self.X_train, self.y_train)
        self.save_model(self.model)
        for X, y, sequences, label in zip((self.X_train, self.X_dev, self.X_test),
                              (self.y_train, self.y_dev, self.y_test),
                              (self.train_sequences, self.dev_sequences, self.test_sequences),
                              ("train", "dev", "test")):
            y_pred = self.model.predict(X)
            print("Evaluating %s data" % label)
            type_tag = get_types_tag(notypes)
            filename = "%s/%s.%s.tsv" % (self.outdir, label, type_tag)
            #print_sequences(sequences, y_pred, filename)
            #result_obj = get_conll_eval(filename)
            #print_results(result_obj)
            '''if not notypes:
                type_tag = get_types_tag(~notypes)
                filename = "%s/%s.%s.tsv" % (self.outdir, label, type_tag)
                print_sequences(sequences, y_pred, filename, notypes=~notypes)
                result_obj = get_conll_eval(filename)'''
                #print_results(result_obj)

    def convert_predict_input(self,input_file):
        print 'Converting'
        lines = []
        with open(input_file) as fp:
            while True:
                text=fp.readline()
                if text=="":
                    break
                lines.append(text)
        converted_lines =[]
        flag = None
        word_list = []
        tag_list = []
        for line in lines:
            l = line.strip()
            if len(l) > 0 and str(l).startswith('#begin') or str(l).startswith('#begin'):
                continue

            l = ' '.join(l.split())
            ls = l.split(" ")
            #word_list = []
            #tag_list = []
            if len(ls) >= 11:
                word = ls[0]  # 0
                pos = ls[3]  # 1
                word_list.append(word)
                tag_list.append(pos)
            else:
                pass
        return word_list, tag_list   

    def predict(self,input_file):
        try:
            self.model
        except:
            #No model present, load pre-treained model
            self.model = self.load_model(pkl_model_path)
        #load_model("tweet_model.pkl")
        #total_tokens = []
        #for each_sent in input_sent:
        words, tags = self.convert_predict_input(input_file)
        #print 'Returned'
        #print len(tags)
        #print (words, tags)
        tokens = tokenizeRawTweetText(" ".join(words).lstrip(" "))
        #print 'Tokens are', tokens
        i = 0
        #while i<len(tokens):
        count=tokens.count('.')
        while count>0:
            tokens.remove('.')
            count=count-1
        #print len(tokens)
        #print type(tokens)
        self.load_features()
        predictions = self.model.predict(([self.get_features(tokens)]))
        entities = []
        previous_state = None
        entity_start = None
        #print predictions
        results = ""
        for i in six.moves.range(len(tokens)):
            token = tokens[i]
            #print 'Token is', token
            label = predictions[0][i]
            #print 'Predictions are', predictions
            #print 'Label is',label
            state = label[0]
            #print_sequences(input_sent,predictions,'output.txt')
            if state in ("B", "U") or \
                     (state in ("I", "E") and previous_state not in ("B", "I")):
                entity_start = i
            if state in ("E", "U") or \
               (state in ("B", "I") and (i == len(tokens) - 1 or predictions[0][i + 1][0] not in ("I", "E"))):
                entity_type = label[2:]
                if entity_type is not None:
                    entities.append((entity_start, i + 1, entity_type))
                    entity_start = None
            previous_state = state
            try:
                if token != '/':
                    result = token + " " + tags[i] + " " + label
                else:
                    result = token + ". " + tags[i] + " " + label
            except:
                #print 'In continue', token
                continue
            results = results + result
            results = results + "\n"
            #print 'token', token
            if token == '/':
                #print 'Inside'
                results = results + "\n"

        f_out = open('output.txt','w')
        f_out.write(results)
        #print (entities)
        return entities
    
    def evaluate(self):
        self.notypes = None
        self.model = CRFModel()
        self.model = self.load_model(pkl_model_path)
        for X, y, sequences, label in zip((self.X_train, self.X_dev, self.X_test),
                              (self.y_train, self.y_dev, self.y_test),
                              (self.train_sequences, self.dev_sequences, self.test_sequences),
                              ("train", "dev", "test")):
            y_pred = self.model.predict(X)
            print("Evaluating %s data" % label)
            type_tag = get_types_tag(self.notypes)
            filename = "%s/%s.%s.tsv" % (self.outdir, label, type_tag)
            print_sequences(sequences, y_pred, filename)
        #list = 
        result_obj = get_conll_eval(filename)

'''train_files = ["/home/khadutz95/TwitterNER/data/cleaned/train.BIEOU.tsv"]
dev_files = ["/home/khadutz95/TwitterNER/data/cleaned/dev.BIEOU.tsv"]
test_files = ["/home/khadutz95/TwitterNER/data/cleaned/test.BIEOU.tsv"]
vocab_file = "./vocab.no_extras.txt"
outdir = "./test_exp"
file_dict = {'train':train_files,'dev':dev_files, 'test':test_files}
# wordvec_file = "/home/entity/Downloads/GloVe/glove.twitter.27B.200d.txt.processed.txt"
wordvec_file = "./data/glove.twitter.27B/glove.twitter.27B.200d.txt"
wordvec_file_processed = "./data/glove.twitter.27B/glove.twitter.27B.200d.txt.processed.txt"
dictionary_dir="./data/cleaned/custom_lexicons/"
gimple_twitter_brown_clusters_dir="./50mpaths2"
data_brown_cluster_dir="brown_clusters_wnut_and_hege/"
data_clark_cluster_dir="clark_clusters_wnut_and_hege/"
enriched_brown_cluster_dir = main_dir_path + "/brown_clusters_wnut_and_hege"
enriched_clark_cluster_dir = main_dir_path+ "/clark_clusters_wnut_and_hege"

twitterner = TwitterNER()
twitterner.read_dataset(outdir,file_dict, vocab_file, False, dataset='WNUT')
twitterner.train()
twitterner.predict('/home/khadutz95/Ner_test_input.txt') 
twitterner.evaluate()'''
wordvec_file = "./data/glove.twitter.27B/glove.twitter.27B.200d.txt"
wordvec_file_processed = "./data/glove.twitter.27B/glove.twitter.27B.200d.txt.processed.txt"
dictionary_dir="./data/cleaned/custom_lexicons/"
gimple_twitter_brown_clusters_dir="./50mpaths2"
data_brown_cluster_dir="brown_clusters_wnut_and_hege/"
data_clark_cluster_dir="clark_clusters_wnut_and_hege/"
enriched_brown_cluster_dir = main_dir_path + "/brown_clusters_wnut_and_hege"
enriched_clark_cluster_dir = main_dir_path+ "/clark_clusters_wnut_and_hege"
