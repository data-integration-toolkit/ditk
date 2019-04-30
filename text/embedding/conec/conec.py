#URL to Parent Class : https://github.com/sandiexie-USC/spring19_csci548_textembedding/blob/master/text_embedding.py
from __future__ import unicode_literals, division, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object, range, next
from glob import glob
import pickle as pkl
import re
import logging
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import LogisticRegression as logreg
from unidecode import unidecode
import math
from scipy import spatial, stats
import nltk

import text_embedding
import word2vec
import context2vec
# from conecNER import ContextEnc_NER

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Text8Corpus(object):
    """
        Task - reads the text8 training Dataset which is present in the "fileName" path
        
        Input:
        self -- conec object which calls this function
        fileName -- string -- Directory Path to the dataset

        return:
        stores the dataset in method appropriate format in self.sentences
    """
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], '', 1000
        with open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split())  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                # the last token may have been split in two... keep it for the next iteration
                last_token = text.rfind(' ')
                words, rest = (text[:last_token].split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]

class OneBilCorpus(object):
    """
        Task - reads the One Billion Words corpus training dataset which is present in the "fileName" path
        
        Input:
        self -- conec object which calls this function
        fileName -- string -- Directory Path to the dataset

        return:
        stores the dataset in method appropriate format in self.sentences
        """
    """Iterate over sentences from the "1-billion-word-language-modeling-benchmark" corpus,
    downloaded from http://code.google.com/p/1-billion-word-language-modeling-benchmark/ ."""

    def __init__(self):
        self.dir = 'data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news*'

    def __iter__(self):
        # go file by file
        for fname in glob(self.dir):
            with open(fname) as f:
                yield f.read().lower().split()

def clean_conll2003(text, to_lower=False):
    # clean the text: no fucked up characters
    text = unidecode(text)
    # normalize numbers
    text = re.sub(r"[0-9]", "1", text)
    if to_lower:
        text = text.lower()
    return text


class CoNLL2003(object):
    # collected 20102 unique words from a corpus of 218609 words and 946 sentences
    # generator for the conll2003 training data

    def __init__(self, to_lower=False, sources=["data/conll2003/train.txt"]):
        self.sources = ["data/conll2003/train.txt"]
        self.to_lower = to_lower

    def __iter__(self):
        """Iterate through all news articles."""
        for fname in self.sources:
            # print(fname)
            tokens = []
            for line in open(fname):
                if line.startswith("-DOCSTART- -X- -X-"):
                    if tokens:
                        yield tokens
                    tokens = []
                elif line.strip():
                    tokens.append(clean_conll2003(line.split()[0], self.to_lower))
                else:
                    tokens.append('')
            yield tokens

class conec(text_embedding.TextEmbedding):
    """
    A child class which implements context encoding for generating text embeddings.
    One of the methods for improving the generated Text Embeddings
    Inherits from the parent class ditk_TextEmbedding

    BENCHMARKS:
        -----------------------------------------------------------------------------------------------------------------------------------------------
        |   DATASET             |         FORMAT                    |               EXAMPLE                      |    EVALUATION METRICS               |     |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | Cornell Movie Reviews | reviews and its sentiment         |"uncompromising french director robert      |   Precision, Recall, F1             |
        | Sentiment Analysis    |                                   | bresson's " lancelot of the lake...", pos  |                                     |     |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | CoNll2003: NER        | entity and its type               | ["LOC","Afghanistan"]                      |   Precision, Recall, F1             |     |
        |----------------------------------------------------------------------------------------------------------------------------------------------- 
        | CategoricalDataset    | data and its category             | ["Office Services Coordinator", 69222.18]  |   Mean Square Error                 |     |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | SemEval: Similarity   | sentences and its similarity score|['Dogs are fighting','Dogs are wrestling',4]|   Pearson Correlation Coefficient   |          | 
        |                       |                                   |                                            |                                     |                                   Coefficient                |
        |-----------------------------------------------------------------------------------------------------------------------------------------------
        | SICK Dataset          | sentences and its similarity score|['Dogs are fighting','Dogs are wrestling',4]| Pearson Correlation Coefficient     |
        |                       |                                   |                                            |                                     |
        ------------------------------------------------------------------------------------------------------------------------------------------------
    
    """

    def __init__(self):
        """
        Constructor which gets initialized when a conec object is declared
        Defines the default data members and sets them to empty strings

        self.dataset -- string -- default dataset name, can be benchmarks or method specific or a generic corpus
        self.dataset_path -- string -- default directory path where the training dataset is stored
        self.model_path -- string -- default directory path where the trained model is stored
        self.result_path -- string -- default directory path where the results of the evaluation dataset is stored
        """
        print("Conec object created ... \nUse this object to get text embeddings or text similarity...")
        self.dataset = ""
        self.dataset_path = "" 
        self.model_path = ""
        self.result_path = ""
        self.model = None
        text_embedding.TextEmbedding.__init__(self)   
    
    def read_Dataset(self, name="text8", path="data/text8"):
        """
        Abstract Function Implementation
        Task -- Reads the input dataset and stores it in self.sentences as a list of tokens. 


        Input:
        self -- conec object which calls this function
        name -- name of the dataset (default text8) (Allowed Arguments: conll2003, cornellMovieReviews, categorical, text8, oneBillionWords)
        path -- path to the dataset

        Action:
        Calls the parent class function if name is a benchmark dataset,
        else calls sub-class function

        Result:
        self.sentences -- list of strings -- Populated with tokens extracted from the dataset
        self.dataset_path -- string -- Populated with the dataset path
        self.model_path -- string -- Populated with a default directory path where the trained model will be stored. This is computed based on the dataset name
        self.result_path -- string -- Populated with a default directory path where the evaluation results will be stored. This is computed based on the dataset name
        self.benchmark_flag -- boolean - Sets it to 1 if name in self.benchmarks, else no change 
        """
        print("Reading dataset ....")
        if name == "text8":
            self.sentences = Text8Corpus(path)
        elif name == "OneBilCorpus":
            self.sentences = OneBilCorpus(path)
        elif name in ['cornellMD', 'conll2003', 'semeval', 'categorical']:
            self.is_benchmark = True
            if name == 'conll2003':
                self.sentences = CoNLL2003(to_lower="False", sources=path)
        else:
            with open(path) as fname:
                self.sentences = fname.read().lower().split()
        self.dataset = name
        self.dataset_path = path
        print("Dataset loaded into conec object ....")

    def train(self, train_all=False,seed=3, iterations=1, saveInterm=False, modelType="cbow", embed_dim=200, minCount=5, window=5, hs=0, neg=13, thr=0, alpha=0.005, min_alpha=0.005, forward=True, backward=True, progress=1000, fill_diag=True, normalize=False): 
        """
        Abstract Function Implementation
        Task -- Train a context vector model on the loaded dataset and save the trained model as a pickle file in the user specfied directory path
        
        Input:
        self -- conec object which calls this function
        modelType -- string -- (default 'sg') type of model: either 'sg' (skipgram) or 'cbow' (bag of words)
        embed_dim -- int -- (default 100) dimensionality of embedding
        hs -- int --  (default 1) if != 0, hierarchical softmax will be used for training the model
        neg -- int -- (default 0) if > 0, negative sampling will be used for training the model;
        neg specifies the # of noise words
        thr -- int -- (default 0) threshold for computing probabilities for sub-sampling words in training
        window -- int -- (default 5) max distance of context words from target word in training
        minCount -- int --  (default 5) how often a word has to occur at least to be taken into the vocab
        alpha -- float -- (default 0.025) initial learning rate
        min_alpha -- float -- (default 0.0001) if < alpha, the learning rate will be decreased to min_alpha
        seed -- int -- (default 1) random seed (for initializing the embeddings) 
        iterations -- int -- (default 3) no of times this model is to be trained on the given dataset and specified parameters
        saveInterm -- boolean -- (default False) Save intermediate model after every iteration to the models/ directory 
        forward -- boolean -- forward pass in the neural network
        backward -- boolean -- backward pass in the neural network
        normalize -- boolean -- (default False) for normalizing the context vector
        fill_diag -- boolean -- (default True) fill diagonal of the context matrix
        progress -- int -- (default 1000) after how many sentences a progress printout should occur
        path -- string -- (default self.model_path) directory path where the trained model will be stored   

        Result:
        Saves the trained model after the final iteration into the user specified path or the default path
        """
        print("conec object being trained using {} dataset present in the directory {} ....".format(self.dataset, self.dataset_path))
        self.model_path = "data/{}_{}_{}_hs{}_neg{}_seed{}_it{}.model".format(self.dataset, modelType, embed_dim, hs, neg, seed, iterations)

        if self.dataset == 'conll2003':
            min_count = 1
            alpha = 0.02
            seed = 3
            iterations = 2
        else:
            min_count = 5

        def save_model(model):
            # delete the huge stupid table again
            table = deepcopy(model.table)
            model.table = None
            # pickle the entire model to disk, so we can load&resume training later
            pkl.dump(model, open(self.model_path, 'wb'), -1)
            # reinstate the table to continue training
            model.table = table

        print("Creating word2vec model ....")
        model = word2vec.Word2Vec(self.sentences, mtype='cbow', hs=0, neg=13, alpha=alpha, min_alpha=alpha, embed_dim=200, seed=3, min_count=min_count)
        print("Training the word2vec model on multiple iterations ....")
        for i in range(1, iterations):
            print("######### ITERATION %i #########" % i)
            if self.dataset == 'conll2003':
                if not i % 5:
                    save_model(model)
                    alpha /= 2.
                    alpha = max(alpha, 0.0001)
            if saveInterm:
                model = save_model(model)
            model.train(self.sentences, alpha=alpha, min_alpha=min_alpha)
        save_model(model)

        if self.dataset == 'conll2003' and train_all:
            sentences = CoNLL2003(to_lower=True, sources=[
                              "data/conll2003/train.txt", "data/conll2003/testa.txt", "data/conll2003/testb.txt"])
            model = word2vec.Word2Vec(sentences, min_count=1, mtype='cbow', hs=0, neg=13, embed_dim=200, seed=seed)
            for i in range(19):
                model.train(sentences)
            # delete the huge stupid table again
            model.table = None
            # pickle the entire model to disk, so we can load&resume training later
            # saven = "conll2003_test_20it_cbow_200_hs0_neg13_seed%i.model" % seed
            print("saving word2vec model trained on CoNLL 2003 dataset into the following file - {}".format(self.model_path))
            pkl.dump(model, open("data/%s" % self.model_path, 'wb'), -1)        
        else:
            with open(self.model_path, 'rb') as f:
                model = pkl.load(f)

            context_model = context2vec.ContextModel(self.sentences, min_count=model.min_count, window=model.window, wordlist=model.index2word)
            context_mat = context_model.get_context_matrix(fill_diag=fill_diag,norm=normalize)
            model.syn0norm = context_mat.dot(model.syn0norm)
            model.syn0norm = model.syn0norm / np.array([np.linalg.norm(model.syn0norm, axis=1)]).T
            print("Saving context2vec model trained on {} dataset in the following file - {}".format(self.dataset, self.model_path))
            save_model(model)
        print("Model for {} dataset saved as {} in the data/ folder".format(self.dataset_path, self.model_path))
    
    def ne_results_2_labels(self, ne_results):
        """
        helper function to transform a list of substrings and labels
        into a list of labels for every (white space separated) token
        """
        l_list = []
        last_l = ''
        for i, (substr, l) in enumerate(ne_results):
            if substr == ' ':
                continue
            if not l or l == 'O':
                l_out = 'O'
            elif l == last_l:
                l_out = "B-" + l
            else:
                l_out = "I-" + l
            last_l = l
            if (not i) or (substr.startswith(' ') or ne_results[i - 1][0].endswith(' ')):
                l_list.append(l_out)
            # if there is no space between the previous and last substring, first token gets label
            # of longer subsubstr (i.e. either previous or current)
            elif i and len(ne_results[i - 1][0].split()[-1]) < len(substr.split()[0]):
                l_list.pop()
                l_list.append(l_out)
            l_list.extend([l_out for n in range(len(substr.split()) - 1)])
        return l_list

    def evaluate(self, model="", filename="", evaluation_type="analogy"):
        """
        Abstract Function Implementation
        Task -- evaluate the provided the trained model on a specified test dataset and store the results in the specified directory path
        
        Input:
        self -- conec object which calls this function
        model -- string -- (default self.model_path) Directory path of the pre-trained model or user-specified model
        filename -- string -- (default self.dataset_path) Directory path for the test dataset
        evaluation_type -- string -- (default "analogy") Can be "analogy", "semeval", "categorical"

        Result:
        The evaluated results for the test dataset on the pre-trained model is stored in the pre-defined directory path
        """
        model = self.model_path
        filename = "results.txt"
        if evaluation_type == "analogy":
            results = self.evaluate_analogy(model, filename)
        elif evaluation_type == "ner":
            results = self.evaluate_ner(model, filename)
        elif evaluation_type == "semeval":
            results = self.evaluate_semeval(model, filename)
        elif evaluation_type == "categorical":
            resutls = self.evaluate_categorical(model, filename)
        else:
            print("Invalid Evaluation Type!\n Please enter one of the following :\n ['ner', 'analogy', 'semeval', 'categorical']")
        return results

    def analogy(self, model, a, b, c):
        # man:woman as king:x - a:b as c:x - find x
        # get embeddings for a, b, and c and multiply with all other words
        a_sims = 1. + np.dot(model.syn0norm, model.syn0norm[model.vocab[a].index])
        b_sims = 1. + np.dot(model.syn0norm, model.syn0norm[model.vocab[b].index])
        c_sims = 1. + np.dot(model.syn0norm, model.syn0norm[model.vocab[c].index])
        # add/multiply them as they should
        return b_sims - a_sims + c_sims
        # return (b_sims*c_sims)/a_sims

    def evaluate_analogy(self, model_path="data/text8_cbow_200_hs0_neg13_seed3_it1.model", output_path = "data/results_analogy.txt", input_path="data/questions-words.txt", restrict_vocab=30000):
        """
        Task -- evaluate the provided the trained model on "analogy" task for a specified test dataset and store the results in the specified directory path
        
        Input:
        self -- conec object which calls this function
        model_path -- string -- (default self.model_path) Directory path of the pre-trained model or user-specified model
        input_path -- string -- (default self.dataset_path) Directory path for the test dataset
        output_path -- string -- (default self.result_path) Directory path for storing the results of the analogy evaluation task
        
        Result:
        The "analogy" task based evaluation results for the test dataset on the pre-trained model is stored in the pre-defined directory path
        
        """
        lowercase=True
        questions = 'data/questions-words.txt'
        with open(model_path, 'rb') as f:
            model = pkl.load(f)
            
        ok_vocab = dict(sorted(model.vocab.items(), key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in ok_vocab.values())

        def log_accuracy(section):
            correct, incorrect = section['correct'], section['incorrect']
            if correct + incorrect > 0:
                print("%s: %.1f%% (%i/%i)" % (section['section'],
                                              100.0 * correct / (correct + incorrect), correct, correct + incorrect))

        sections, section = [], None
        for line_no, line in enumerate(open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    if lowercase:
                        a, b, c, expected = [word.lower() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except:
                    print("skipping invalid line #%i in %s" % (line_no, questions))
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    # print "skipping line #%i with OOV words: %s" % (line_no, line)
                    continue

                ignore = set(model.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                # for index in np.argsort(model.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
                for index in np.argsort(self.analogy(model, a, b, c))[::-1]:
                    if index in ok_index and index not in ignore:
                        predicted = model.index2word[index]
                        # if predicted != expected:
                        #     print "%s: expected %s, predicted %s" % (line.strip(), expected, predicted)
                        break
                section['correct' if predicted == expected else 'incorrect'] += 1
        if section:
            # store the last section, too
            sections.append(section)
            log_accuracy(section)

        total = {'section': 'total', 'correct': sum(s['correct']
                                                    for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
        log_accuracy(total)
        sections.append(total)
        return sections    

    def apply_conll2003_ner(self, ner, testfile, outfile):
        """
        Inputs:
            - ner: named entity classifier with find_ne_in_text method
            - testfile: path to the testfile
            - outfile: where the output should be saved
        """
        documents = CoNLL2003(sources=[testfile], to_lower=True)
        documents_it = documents.__iter__()
        local_context_mat, tok_idx = None, {}
        # read in test file + generate outfile
        with open(outfile, 'w') as f_out:
            # collect all the words in a sentence and save other rest of the lines
            to_write, tokens = [], []
            doc_tokens = []
            for line in open(testfile):
                if line.startswith("-DOCSTART- -X- -X-"):
                    f_out.write("-DOCSTART- -X- -X- O O\n")
                    # we're at a new document, time for a new local context matrix
                    if ner.context_model:
                        doc_tokens = next(documents_it)
                        local_context_mat, tok_idx = ner.context_model.get_local_context_matrix(doc_tokens)
                # outfile: testfile + additional column with predicted label
                elif line.strip():
                    to_write.append(line.strip())
                    tokens.append(clean_conll2003(line.split()[0]))
                else:
                    # end of sentence: find all named entities!
                    if to_write:
                        ne_results = ner.find_ne_in_text(" ".join(tokens), local_context_mat, tok_idx)
                        assert " ".join(tokens) == "".join(r[0]
                                                           for r in ne_results), "returned text doesn't match"  # sanity check
                        l_list = self.ne_results_2_labels(ne_results)
                        assert len(l_list) == len(tokens), "Error: %i labels but %i tokens" % (len(l_list), len(tokens))
                        for i, line in enumerate(to_write):
                            f_out.write(to_write[i] + " " + l_list[i] + "\n")
                    to_write, tokens = [], []
                    f_out.write("\n")


    def log_results(self, clf_ner, description, filen='', subf=''):
        import os
        if not os.path.exists('data/conll2003_results'):
            os.mkdir('data/conll2003_results')
        if not os.path.exists('data/conll2003_results%s' % subf):
            os.mkdir('data/conll2003_results%s' % subf)
        import subprocess
        print("applying to training set")
        self.apply_conll2003_ner(clf_ner, 'data/conll2003/train.txt', 'data/conll2003_results%s/out_train.txt' % subf)
        print("applying to test set")
        self.apply_conll2003_ner(clf_ner, 'data/conll2003/testa.txt', 'data/conll2003_results%s/out_testa.txt' % subf)
        self.apply_conll2003_ner(clf_ner, 'data/conll2003/testb.txt', 'data/conll2003_results%s/out_testb.txt' % subf)
        # write out results
        with open('data/conll2003_results/output_all_%s.txt' % filen, 'a') as f:
            f.write('%s\n' % description)
            f.write('results on training data\n')
            out = subprocess.getstatusoutput('data/conll2003/conlleval < data/conll2003_results%s/out_train.txt' % subf)[1]
            f.write(out)
            f.write('\n')
            f.write('results on testa\n')
            out = subprocess.getstatusoutput('data/conll2003/conlleval < data/conll2003_results%s/out_testa.txt' % subf)[1]
            f.write(out)
            f.write('\n')
            f.write('results on testb\n')
            out = subprocess.getstatusoutput('data/conll2003/conlleval < data/conll2003_results%s/out_testb.txt' % subf)[1]
            f.write(out)
            f.write('\n')
            f.write('\n')    

    def evaluate_ner(self, model_path="data/conll2003_train_cbow_200_hs0_neg13_seed3_it20.model", input_path="data/conll2003/testa.txt", output_path="data/results_ner.txt"):
        """
        Task -- evaluate the provided the trained model on "ner" task for a specified test dataset and store the results in the specified directory path
        
        Input:
        self -- conec object which calls this function
        model_path -- string -- (default self.model_path) Directory path of the pre-trained model or user-specified model
        input_path -- string -- (default self.dataset_path) Directory path for the test dataset
        output_path -- string -- (default self.result_path) Directory path for storing the results of the analogy evaluation task
        
        Result:
        The "ner" task based evaluation results for the test dataset on the pre-trained model is stored in the pre-defined directory path
        """
        with open(self.model_path, 'rb') as f:
            model = pkl.load(f)

        clf_ner = ContextEnc_NER(model, include_wf=False)
        clf_ner.train_clf(['data/conll2003/train.txt'])
        
        seed = 3
        it = 20

        self.log_results(clf_ner, '####### word2vec model, seed: %i, it: %i' % (seed, it), 'word2vec_%i' % seed, '_word2vec_%i_%i' % (seed, it))

        sentences = CoNLL2003(to_lower=True)
        # only use global context; no rep for out-of-vocab
        clf_ner = ContextEnc_NER(model, contextm=True, sentences=sentences, w_local=0., context_global_only=True)
        clf_ner.train_clf(['data/conll2003/train.txt'])
        # evaluate the results again
        self.log_results(clf_ner, '####### context enc with global context matrix only, seed: %i, it: %i' % (seed, it), 'conec_global_%i' % seed, '_conec_global_%i_%i' % (seed, it))

        # for the out-of-vocabulary words in the dev and test set, only the local context matrix (based on only the current doc)
        # is used to generate the respective word embeddings; where a global context vector is available (for all words in the training set)
        # we use a combination of the local and global context, determined by w_local
        for w_local in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
            print(w_local)
            clf_ner = ContextEnc_NER(model, contextm=True, sentences=sentences, w_local=w_local)
            clf_ner.train_clf(['data/conll2003/train.txt'])
            # evaluate the results again
            self.log_results(clf_ner, '####### context enc with a combination of the global and local context matrix (w_local=%.1f), seed: %i, it: %i' % (w_local, seed, it), 'conec_%i_%i' % (round(w_local*10), seed), '_conec_%i_%i_%i' % (round(w_local*10), seed, it))


    def evaluate_categorical(self, model_path, input_path, output_path, lowercase=True, restrict_vocab=30000):
        """
        Task -- evaluate the provided the trained model on "categorical" task for a specified test dataset and store the results in the specified directory path
        
        Input:
        self -- conec object which calls this function
        model_path -- string -- (default self.model_path) Directory path of the pre-trained model or user-specified model
        input_path -- string -- (default self.dataset_path) Directory path for the test dataset
        output_path -- string -- (default self.result_path) Directory path for storing the results of the analogy evaluation task
        
        Result:
        The "categorical" task based evaluation results for the test dataset on the pre-trained model is stored in the pre-defined directory path
        """
        pass

    def evaluate_semeval(self, model_path, input_path, output_path, lowercase=True, restrict_vocab=30000):
        """
        Task -- evaluate the provided the trained model on "semeval" task for a specified test dataset and store the results in the specified directory path
        
        Input:
        self -- conec object which calls this function
        model_path -- string -- (default self.model_path) Directory path of the pre-trained model or user-specified model
        input_path -- string -- (default self.dataset_path) Directory path for the test dataset
        output_path -- string -- (default self.result_path) Directory path for storing the results of the analogy evaluation task
        
        Result:
        The "semeval" task based evaluation results for the test dataset on the pre-trained model is stored in the pre-defined directory path
        """
        pass

    def evaluate_sick(self, model_path="data/text8_cbow_200_hs0_neg13_seed3_it1.model", input_path="data/sick/SICK.txt", output_path="data/sick/results.txt"):
        """
        Task -- evaluate the provided the trained model on "semeval" task for a specified test dataset and store the results in the specified directory path
        
        Input:
        self -- conec object which calls this function
        model_path -- string -- (default self.model_path) Directory path of the pre-trained model or user-specified model
        input_path -- string -- (default self.dataset_path) Directory path for the test dataset
        output_path -- string -- (default self.result_path) Directory path for storing the results of the analogy evaluation task
        
        Result:
        The "semeval" task based evaluation results for the test dataset on the pre-trained model is stored in the pre-defined directory path
        """
        # reader = ModelReader(model_file)
        f = open(input_path, 'r')
        dataset = f.readlines()

        model_path = "data/text8_cbow_200_hs0_neg13_seed3_it1.model"

        with open(model_path, 'rb') as f:
            model = pkl.load(f)

        def get_avg(input):

            vectors = [model.syn0norm[model.vocab[t.lower()].index]
                       for t in nltk.word_tokenize(input)]
            vectors = [v for v in vectors if v is not None]
            if not vectors:
                return None

            ret = np.mean(vectors, axis=0)
            # ret = np.dot(ret, self._W)
            # ret += self._b

            ret /= np.linalg.norm(ret, 2)
            return ret

        predicted = []
        correct = []
        x = open(output_path, 'w')
        for (n, line) in enumerate(dataset):
            if n == 0:
                continue

            data = line.rstrip().split('\t')
            sent1 = data[1]
            sent2 = data[2]
            score = float(data[4])
            fold = data[11]
            if fold == 'TRIAL':
                continue

            correct.append(float(score))

            vec1 = get_avg(sent1)
            vec2 = get_avg(sent2)
            predicted.append(1.0 - spatial.distance.cosine(vec1, vec2))
        x.write('%.4f (pearson) %.4f (spearman)' % (pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0]))


    def predict_embedding(self, inp):
        """
        Task - Predicts either based on the pre-trained model or the custom trained model depending on the benchmark flag and returns the best embedding for the token in the input list

        Input: 
        self -- conec object which calls this function
        input -- list of strings -- List of token(s)

        Action - Predicts the embedding vector for a string or an average vector for a sentence/corpus based on the pre-trained or custom-trained model

        return: 
        embedding -- vector -- a vector embedding for the token/ list of tokens
        """
        # if self.model_path == "":
        model_path="data/text8_cbow_200_hs0_neg13_seed3_it1.model"
        # else:
            # model_path = self.model_path
        with open(model_path, 'rb') as f:
            model = pkl.load(f)        
       
        return model.syn0norm[model.vocab[inp].index]

    def predict_sent_embedding(self, inp):
        """
        Task - Predicts either based on the pre-trained model or the custom trained model depending on the benchmark flag and returns the best embedding for the token in the input list

        Input: 
        self -- conec object which calls this function
        input -- list of strings -- List of token(s)

        Action - Predicts the embedding vector for a string or an average vector for a sentence/corpus based on the pre-trained or custom-trained model

        return: 
        embedding -- vector -- a vector embedding for the token/ list of tokens
        """
        return self.get_average_embedding(inp)

    def predict_similarity(self, input1, input2):
        """
        Task - Predicts either based on the pre-trained model or the custom trained model depending on the benchmark flag and returns the wmd similairty between two embedding vectors

        Input: 
        self -- conec object which calls this function
        input1 -- list of strings -- List of token(s) representing input1
        input2 -- list of strings -- List of token(s) representing input2

        Action - Calculates the embeddings/average embeddings for the two inputes and computes the similarity between them

        return: 
        similarity score -- float -- word mover distance from gensim
        """
        if self.model_path == "":
            model_path="data/text8_cbow_200_hs0_neg13_seed3_it1.model"
        else:
            model_path = self.model_path

        with open(model_path, 'rb') as f:
            model = pkl.load(f)
        return np.inner(model.syn0norm[model.vocab[input1].index], model.syn0norm[model.vocab[input2].index])

    def predict_sent_similarity(self, input1, input2):
        """
        Task - Predicts either based on the pre-trained model or the custom trained model depending on the benchmark flag and returns the wmd similairty between two embedding vectors

        Input: 
        self -- conec object which calls this function
        input1 -- list of strings -- List of token(s) representing input1
        input2 -- list of strings -- List of token(s) representing input2

        Action - Calculates the embeddings/average embeddings for the two inputes and computes the similarity between them

        return: 
        similarity score -- float -- word mover distance from gensim
        """
        if self.model_path == "":
            model_path="data/text8_cbow_200_hs0_neg13_seed3_it1.model"
        else:
            model_path = self.model_path

        with open(model_path, 'rb') as f:
            model = pkl.load(f)
        inp1_vec = self.get_average_embedding(input1)
        inp2_vec = self.get_average_embedding(input2)
        return np.inner(inp1_vec, inp2_vec)


    def get_average_embedding(self, input):
        """
        Task - Calculates the average embedding over all the input embeddings for ex. for all words in a sentence

        Input: 
        self -- conec object which calls this function
        input -- list of vectors -- List of embeddings which represents words in a sentence or a paragraph/corpus

        Action - Calculates the average embeddings over all the input embeddings

        return: 
        average_embedding -- vector -- average embedding over all the word embeddings in a sentence/paragraph
        """
        if self.model_path == "":
            model_path="data/text8_cbow_200_hs0_neg13_seed3_it1.model"
        else:
            model_path = self.model_path

        with open(model_path, 'rb') as f:
            model = pkl.load(f)

        vectors = [model.syn0norm[model.vocab[t.lower()].index]
                   for t in input.split()]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return None

        ret = np.mean(vectors, axis=0)
        # ret = np.dot(ret, self._W)
        # ret += self._b

        ret /= np.linalg.norm(ret, 2)

        return ret


    def save_model(self, model, model_path):
        """
        Task -- Saves the trained model into the specfied path

        Input:
        self -- conec object which calls this function
        model -- matrix -- train model which is to be saved (from the training function)
        model_path -- string -- (default self.model_path) directory path where this trained model must be stored as a pickle dump

        Result:
        Trained model is saved in the specified model_path directory
        """
        #delete the huge table again
        self.model_path = "data/" + model_path + ".model"
        table = deepcopy(model.table)
        model.table = None
        #pickle the entire model to disk, so we can load & resume training later
        pkl.dump(model, open(self.model_path, 'wb'), -1)
        #reinstate the table to continue training
        model.table = table
        return model

    def load_model(self, file):
        self.model_path = file

class ContextEnc_NER(object):

    def __init__(self, w2v_model, contextm=False, sentences=[], w_local=0.4, context_global_only=False, include_wf=False, to_lower=True, normed=True, renorm=True):
        self.clf = None
        self.w2v_model = w2v_model
        self.rep_idx = {word: i for i, word in enumerate(w2v_model.index2word)}
        self.include_wf = include_wf
        self.to_lower = to_lower
        self.w_local = w_local  # how much the local context compared to the global should count
        self.context_global_only = context_global_only  # if only global context should count (0 if global not available -- not same as w_local=0)
        self.normed = normed
        self.renorm = renorm
        # should we include the context?
        if contextm:
            # sentences: depending on what the word2vec model was trained
            self.context_model = context2vec.ContextModel(
                sentences, min_count=1, window=w2v_model.window, wordlist=w2v_model.index2word)
            # --> create a global context matrix
            self.context_model.featmat = self.context_model.get_context_matrix(False, 'max')
        else:
            self.context_model = None

    def make_featmat_rep(self, tokens, local_context_mat=None, tok_idx={}):
        """
        Inputs:
            - tokens: list of words
        Returns:
            - featmat: dense feature matrix for every token
        """
        # possibly preprocess tokens
        if self.to_lower:
            pp_tokens = [t.lower() for t in tokens]
        else:
            pp_tokens = tokens
        dim = self.w2v_model.syn0norm.shape[1]
        if self.include_wf:
            dim += 7
        featmat = np.zeros((len(tokens), dim), dtype=float)
        # index in featmat for all known tokens
        idx_featmat = [i for i, t in enumerate(pp_tokens) if t in self.rep_idx]
        if self.normed:
            rep_mat = deepcopy(self.w2v_model.syn0norm)
        else:
            rep_mat = deepcopy(self.w2v_model.syn0)
        if self.context_model:
            if self.context_global_only:
                # make context matrix out of global context vectors only
                context_mat = lil_matrix((len(tokens), len(self.rep_idx)))
                global_tok_idx = [self.rep_idx[t] for t in pp_tokens if t in self.rep_idx]
                context_mat[idx_featmat, :] = self.context_model.featmat[global_tok_idx, :]
            else:
                # compute the local context matrix
                if not tok_idx:
                    local_context_mat, tok_idx = self.context_model.get_local_context_matrix(pp_tokens)
                local_tok_idx = [tok_idx[t] for t in pp_tokens]
                context_mat = lil_matrix(local_context_mat[local_tok_idx, :])
                assert context_mat.shape == (len(tokens), len(self.rep_idx)), "context matrix has wrong shape"
                # average it with the global context vectors if available
                local_global_tok_idx = [tok_idx[t] for t in pp_tokens if t in self.rep_idx]
                global_tok_idx = [self.rep_idx[t] for t in pp_tokens if t in self.rep_idx]
                context_mat[idx_featmat, :] = self.w_local * lil_matrix(local_context_mat[local_global_tok_idx, :]) + (
                    1. - self.w_local) * self.context_model.featmat[global_tok_idx, :]
            # multiply context_mat with rep_mat to get featmat (+ normalize)
            featmat[:, 0:rep_mat.shape[1]] = csr_matrix(context_mat) * rep_mat
            # length normalize the feature vectors
            if self.renorm:
                fnorm = np.linalg.norm(featmat, axis=1)
                featmat[fnorm > 0, :] = featmat[fnorm > 0, :] / np.array([fnorm[fnorm > 0]]).T
        else:
            # we set the feature matrix with the word2vec embeddings directly;
            # tokens not in the original vocab will have a zero representation
            idx_repmat = [self.rep_idx[t] for t in pp_tokens if t in self.rep_idx]
            featmat[idx_featmat, 0:rep_mat.shape[1]] = rep_mat[idx_repmat, :]
        if self.include_wf:
            featmat[:, dim - 7:] = make_featmat_wordfeat(tokens)
        return featmat

    def train_clf(self, trainfiles):
        # tokens: list of words, labels: list of corresponding labels
        # go document by document because of local context
        final_labels = []
        featmat = []
        for trainfile in trainfiles:
            for tokens, labels in yield_tokens_labels(trainfile):
                final_labels.extend(labels)
                featmat.append(self.make_featmat_rep(tokens))
        featmat = np.vstack(featmat)
        print("training classifier")
        clf = logreg(class_weight='balanced', random_state=1)
        clf.fit(featmat, final_labels)
        self.clf = clf

    def find_ne_in_text(self, text, local_context_mat=None, tok_idx={}):
        featmat = self.make_featmat_rep(text.strip().split(), local_context_mat, tok_idx)
        labels = self.clf.predict(featmat)
        # stitch text back together
        results = []
        for i, t in enumerate(text.strip().split()):
            if results and labels[i] == results[-1][1]:
                results[-1] = (results[-1][0] + " " + t, results[-1][1])
            else:
                if results:
                    results.append((' ', 'O'))
                results.append((t, labels[i]))
        return results

def make_wordfeat(w):
        return [int(w.isalnum()), int(w.isalpha()), int(w.isdigit()),
            int(w.islower()), int(w.istitle()), int(w.isupper()),
            len(w)]

def make_featmat_wordfeat(tokens):
    # tokens: list of words
    return np.array([make_wordfeat(t) for t in tokens])

def process_wordlabels(word_labels):
    # process labels
    tokens = []
    labels = []
    for word, l in word_labels:
        if word:
            if l.startswith("I-") or l.startswith("B-"):
                l = l[2:]
            tokens.append(word)
            labels.append(l)
    assert len(tokens) == len(labels), "must have same number of tokens as labels"
    return tokens, labels

def get_tokens_labels(trainfile):
    # read in trainfile to generate training labels
    with open(trainfile) as f:
        word_labels = [(clean_conll2003(line.split()[0]), line.strip().split()[-1]) if line.strip()
                       else ('', 'O') for line in f if not line.startswith("-DOCSTART- -X- -X-")]
    return process_wordlabels(word_labels)

def yield_tokens_labels(trainfile):
    # generate tokens and labels for every document
    word_labels = []
    for line in open(trainfile):
        if line.startswith("-DOCSTART- -X- -X-"):
            if word_labels:
                yield process_wordlabels(word_labels)
            word_labels = []
        elif line.strip():
            word_labels.append((clean_conll2003(line.split()[0]), line.strip().split()[-1]))
        else:
            word_labels.append(('', 'O'))
    yield process_wordlabels(word_labels)

def ne_results_2_labels(ne_results):
    """
    helper function to transform a list of substrings and labels
    into a list of labels for every (white space separated) token
    """
    l_list = []
    last_l = ''
    for i, (substr, l) in enumerate(ne_results):
        if substr == ' ':
            continue
        if not l or l == 'O':
            l_out = 'O'
        elif l == last_l:
            l_out = "B-" + l
        else:
            l_out = "I-" + l
        last_l = l
        if (not i) or (substr.startswith(' ') or ne_results[i - 1][0].endswith(' ')):
            l_list.append(l_out)
        # if there is no space between the previous and last substring, first token gets label
        # of longer subsubstr (i.e. either previous or current)
        elif i and len(ne_results[i - 1][0].split()[-1]) < len(substr.split()[0]):
            l_list.pop()
            l_list.append(l_out)
        l_list.extend([l_out for n in range(len(substr.split()) - 1)])
    return l_list

# def main():
#     x = conec()
#     # x.evaluate_ner()
#     x.read_Dataset('conll2003')
#     x.train("cbow_conll2003_model")
#     x.evaluate_ner()
#     # print(x.predict_embedding('m'))
#     # x.read_Dataset('text8', 'data/text8')
#     # x.train("cbow_text8_model", iterations=1)
#     # print(x.predict_similarity("man", "man"))
#     # print(x.get_average_embedding('I am a girl'))
#     # x.evaluate_sick()
#     # print(x.evaluate_analogy('data/text8_cbow_200_hs0_neg13_seed3_it1.model', 'data/results_analogy.txt', 'analogy'))

# if __name__ == '__main__':
#     main()