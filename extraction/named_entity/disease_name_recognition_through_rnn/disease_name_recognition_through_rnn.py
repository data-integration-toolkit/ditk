import abc
import sys
from ner_2 import Ner  # <-- PARENT CLASS POINTER: https://github.com/AbelJJohn/csci548sp19projectner/blob/master/ner.py...NOTE had to mod for python2...
from temp1 import model_train
from data_proc import create_wl, invert_dict
from dataset import Dataset
import drtrnn_utils as dutil
from six.moves import cPickle
import numpy as np
from utils import backtrack

from model1 import *
# import numpy as np
# import theano
# import theano.tensor as T
# import theano.tensor.nnet as nnet

# from emb import *
# from hidden import *
# from output import *
# from trainer import *

sys.setrecursionlimit(5000)

# ---------------------Helper for saving model-------------------- #
import copy_reg
import types
import multiprocessing

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

# -------------------End Helper for saving model------------------ #


"""
Abstraction for interface between DITKModel_NER and the code implementation of paper:
- Sunil Kumar Sahu,et. al. Recurrent neural network models for disease name
    recognition using domain invariant features. Proceedings if the 54th Annual Meeting
    of the Association for Computational Linguistics, pages 2216-2225, 12 Aug 2016
- github repo: https://github.com/suniliggu/disease_name_recognition_through_rnn
"""

class disease_name_recognition_through_rnn(Ner):
    data_path_base = 'binary_data/'
    model_save_filename = 'trained_model.save'
    prediction_filename = 'predictions.txt'

# -------------------------------------init FOR MY CLASS------------------------------#
    def __init__(self,trained_model=None,embeddings=None,other_datas={},**kwargs):
        self.trained_model = trained_model  # nothing to start. filled by train()

        self.n_iter = 50001  # Number of iteration train
        self.m_report = 10000  # after m iterations, predict and report performance on the dev and test sets
        self.n_f = 100  # dim of word embedding
        self.n_hidden = 100  # number of nodes in hidden layer
        self.model = 'rnn'  # model used for classification : 'ff' or 'rnn'
        self.viterbi = True  # structure output or not
        self.trainer = 'AdagradTrainer'  # Training method
        self.lr = 0.05  # learning rate
        self.batch_size = 1  # batch size for minibatch training
        self.wl = None
        self.id2word = None
        self.tl = {"O": 0, "B-Dis": 1, "I-Dis":2}
        self.save_checkpoint_models = False

        # user level control of training parameters [see https://github.com/sunilitggu/disease_name_recognition_through_rnn/blob/master/model1.py]
        self.embeddings = embeddings  # file location to pull embeddings for training [stored in author github] [NOT YET IMPLEMENTED requires changes to emb.py]
        # if 'l_vocab_w' in kwargs:
        #     self.l_vocab_w = kwargs['l_vocab_w']
        # if 'l_vocab_out' in kwargs:
        #     self.l_vocab_out = kwargs['l_vocab_out']
        if 'n_iter' in kwargs:
            self.n_iter = kwargs['n_iter']
        if 'm_report' in kwargs:
            self.m_report = kwargs['m_report']
        if 'n_f' in kwargs:
            self.n_f = kwargs['n_f']
        if 'n_hidden' in kwargs:
            self.n_hidden = kwargs['n_hidden']
        # if 'trainer' in kwargs:
        #     self.trainer = kwargs['trainer']
        # if 'activation' in kwargs:
        #     self.activation = kwargs['activation']
        if 'model' in kwargs:
            if kwargs['model'] in ['ff','rnn']:
                self.model = kwargs['model']  # ff or rnn
        if 'viterbi' in kwargs:
            self.viterbi = kwargs['viterbi']  # viterbi, represent!! ;)
        if 'lr' in kwargs:
            self.lr = kwargs['lr']  # learning rate
        if 'save_checkpoint_models' in kwargs:
            self.save_checkpoint_models = kwargs['save_checkpoint_models']  # learning rate

        # if 'fname' in kwargs:
        #     self.fname = kwargs['fname']
        # if 'wl' in kwargs:
        #     self.wl = kwargs['wl']
        # if 'windim' in kwargs:
        #     self.windim = kwargs['windim']

        self.other_datas = other_datas  # data structure for special model parameters/extra datas

# -------------------------------------END init FOR MY CLASS------------------------------#

# -------------------------------------METHODS FOR MY CLASS------------------------------#


    def tokenize(self, dataInstance, mode):
        """
        Some data conversions may reqiure special tokenization.

        Args:
            dataInstance: some format from intermediate step in convert_dataset_[01,02,03,orif]_to_train_format

        Return:
            tokens: tokenized version of dataInstance

        Raises:
            None
        """
        # TO IMPLEMENT. pseudo code
        # tokens = ntlk.tokenize(dataInstance)
        return tokens


# read_dataset will read the raw files that the user specified and will:
# a) convert the data and rewrite in in format used by paper implementation of train() since the authors
#   implemented a data read into their train() implelemntation
# b) convert the data to the format used by paper implementation of train() and return this data. then, on
#   train(), pass this data and skip the paper's portion of reading the dataset
# EITHER WAY we need to write a conversion helper function to convert the data to the proper format


    def convert_dataset_chemdner_to_train_format(self, dataInstance, mode):
        """
        Converts a single data instance from read dataset_03 to format required by paper implementation.

        Args:
            dataInstance: str, a single line from the raw input data for dataset_03
            mode: if write, convert to proper str format

        Returns:
            data: data format required by author implementation and based on mode

        Raises:
            None
        """
        # IMPLEMENT convert dataInstance to data

        if mode == 'write':
            data = str(data)+'\n'
        return data

    def convert_dataset_ppim_to_train_format(self, dataInstance, mode):
        """
        Converts a single data instance from read dataset_orig to format required by paper implementation.

        Args:
            dataInstance: str, a single line from the raw input data for dataset_orig
            mode: if write, convert to proper str format

        Returns:
            data: data format required by author implementation and based on mode

        Raises:
            None
        """
        # IMPLEMENT convert dataInstance to data

        if mode == 'write':
            data = str(data)+'\n'
        return data


    def convert_author_prediction_to_ditk_ner_prediction(self,predictions):
        """
        Converts the predictions made by author implementation of predict() to the agreed upon format
        used by the ditk.ner group. Helper function in perparing for evaluate()

        Args:
            predictions: iterable in format that author of paper implemented

        Returns:
            converted_predictions: [tuple,...], i.e. list of tuples. list  is same length as input predictions but in
            format agreed upon by team ditk.ner, namely:
                Each tuple is (start index, span, mention text, mention type)
                Where:
                - start index: int, the index of the first character of the mention span. None if not applicable.
                - span: int, the length of the mention. None if not applicable.
                - mention text: str, the actual text that was identified as a named entity. Required.
                - mention type: str, the entity/mention type. None if not applicable.

                NOTE: ordering should not change [important for evalutation. See note in evaluate() about parallel arrays.]

        Raises:
            None
        """
        # IMPLEMENT the proper conversion, pseudo code:
        # converted_predictions = []
        # for prediction in predictions:
        #   converted_item = some splitting and re-arranging
        #   converted_predictions.append(converted_item)
        return converted_predictions
# -----------------------------------END METHODS FOR MY CLASS------------------------------#


# ----------------------------IMPLEMENT PARENT CLASS ABASTRACTMETHODS------------------------------#
    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        DEPRECATED. NO LONGER IN USE. USE dutil.load_groundTruth_from_predictions()


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

        return


    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):  # <--- implemented PER class
        """
        Reads a dataset in preparation for train or test. Returns data in proper format for train or test.
        Also writes datasets in proper format in proper fixed location for train and test.
        Args:
            file_dict: dictionary
                 {
                    "train": dict, {key="file description":value="file location"},  'file description' expected to be str in set {'data',...}
                    "dev" : dict, {key="file description":value="file location"},
                    "test" : dict, {key="file description":value="file location"},
                 }
            PRECONDITION: 'file description' expected to be str in set {'data',...}

            dataset_name: str
                Name of the dataset required for calling appropriate utils, converters
                    must be in {'CoNLL_2003','OntoNotes_5p0','CHEMDNER','unittest','ditk'}
                    If other datasets are required, suggest writing your own converter to 
                    ditk format and use 'ditk' [note, example ditk format found in test/sample_input.txt]
        
        Returns:
            data_dict: dictionary
                {
                    'train': list of lists.
                    'dev': list of lists.
                    'test': list of lists.
                }
            NOTE: list of list. inner list is [token,tag]

        Raises:
            None
        """

        # benchmark_orig represents dataset authors originally used. NOT common to whole NER group
        conversionFunctionMapper = {'CoNLL_2003':dutil.convert_dataset_conll_to_train_format,
            'OntoNotes_5p0':dutil.convert_dataset_ontoNotes_to_train_format,
            'CHEMDNER':dutil.convert_dataset_chemdner_to_train_format,
            'ppim':self.convert_dataset_ppim_to_train_format,
            'unittest':dutil.convert_ditk_to_train_format}

        if not (dataset_name in conversionFunctionMapper):
            print("dataset not supported. Please indicate a dataset in {'CoNLL_2003','OntoNotes_5p0','CHEMDNER','ppim'}")
            return None

        # select proper conversion function based on dataset name
        converter = conversionFunctionMapper[dataset_name]

        # perform the conversion [note files for drtrnn are written with proper filenames and locations]
        data_dict = converter(file_dict)

        return data_dict


    def train(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        Trains a model on the given input data
        Args:
            data: dictionary
                {
                    'train': list of lists.
                    'dev': list of lists.
                    'test': list of lists.
                }
            NOTE: list of list. inner list is [token,tag]
            NOTE: okay for data to be empty or None. In this case, expect a binary_data/train.txt must exist for proper execution
        Returns:
            ret: None. Trained model stored internally to class instance state.
        Raises:
            None
        """
        # IMPLEMENT TRAINING.
        # pass
        # simple call to author implementation of train() expects data already in proper format from conversion
        # something like:
        # self.trained_model = trained model from authors implementation of train(pass data, other train parameters [i.e. self.batch_size, etc...])  <---STORE TRAINED MODEL DATA INTERNAL TO INSTANCE [instance variabel self.trained_model]
        if not (len(data) < 1):  # expect empty iterable. if not, write new training file based on data
            dutil.write_drtrnn_format_to_file(data,self.data_path_base +'train.txt')  # NOT YET TESTED
        
        dutil.bio()  # another preprocessing of input data. generates *_tags and *_words files and the vocabulary

        # generate and save the word counts dict. derived from vocab file, which was generated in bio()
        vocab_file_location = 'word2vec/vocab.txt'
        self.wl = wl = create_wl(vocab_file_location)
        self.id2word = invert_dict(wl)


        # temp1.model_train(N,M,n_f,n_hidden,model,viterbi,trainer,lr,batch_size,wl,tl,save_checkpoint_models=False)
        model = model_train(self.n_iter,self.m_report,self.n_f,self.n_hidden,
            self.model,self.viterbi,self.trainer,self.lr,self.batch_size,
            self.wl,self.id2word,self.tl,save_checkpoint_models=self.save_checkpoint_models)  # train the model!  concurrently predicts on the model..but how to SAVE model for saved use???

        self.trained_model = model

        
        print('Saving trained model to file: %s'%self.model_save_filename)
        self.save_model('trained_model.save')

        return  # None


    def predict(self, data, *args, **kwargs):  # <--- implemented PER class WITH requirement on OUTPUT format!
        """
        Predicts on the given input data. Assumes model has been trained with train()
        Args:
            data: dictionary
                {
                    'train': list of lists.
                    'dev': list of lists.
                    'test': list of lists.
                }
            NOTE: list of list. inner list is [token,tag]
            NOTE: okay for data to be empty or None. In this case, expect
                a binary_data/test.txt must exist for proper execution. Also, expect a trained model exists
            NOTE: if any data samples are single word sentences, they are dropped by paper implementation code

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

        if not (len(data) < 1):  # expect empty iterable. if not, write new testing file based on data
            dutil.write_drtrnn_format_to_file(data,self.data_path_base +'test.txt')  # NOT YET TESTED
            dutil.bio(test_only=True)  # NOT YET IMPLEMENTED....may need to do another round of this.....if JUST data is passed...


        test_w = "test_words.txt"
        test_t = "test_tags.txt"

        testset = Dataset(self.data_path_base+test_w, self.data_path_base+test_t, self.wl, self.tl, self.batch_size, shuffle_samples=False)
        print "testset", testset.tot
        test_sampler = testset.sampler()


        if self.trained_model==None:
            # must load one. else use the one already stored
            print('Loading saved model from file: %s'%self.model_save_filename)
            try:
                self.load_model(self.model_save_filename)
            except:
                print('Error loading saved model file. No predictions made.')
                return None

            
        g = open(self.prediction_filename,"w+")
                #TEST DATA
        predictions = []
        for m in range(testset.tot):
            test_inputs, test_tags = test_sampler.next()        #dev data
            res = self.trained_model.eval_perplexity(test_inputs[0], test_tags[0])
            pred = self.trained_model.predict(test_inputs[0])
            viterbi_max, viterbi_argmax =  self.trained_model.output_decode(test_inputs[0])  
            first_ind = np.argmax(viterbi_max[-1])
            viterbi_pred =  backtrack(first_ind, viterbi_argmax)
            vi_pre = np.array(viterbi_pred)
            test_true = list(test_tags[0])
            for k,l,n in zip(vi_pre, test_true, test_inputs[0]):
                item_predict = [None,None]
                g.write(self.id2word[n])
                item_predict.append(self.id2word[n])
                # print n
                g.write(" ")
                if(l == 0):
                    g.write("O")
                if(l == 1):
                    g.write("B-Dis")
                if(l == 2):
                    g.write("I-Dis")    
                g.write(" ")
                mentionType = ''
                if(k == 0):
                    g.write("O")
                    mentionType = 'O'
                if(k == 1):
                    g.write("B-Dis")
                    mentionType = 'B-Dis'
                if(k == 2):
                    g.write("I-Dis")
                    mentionType = 'I-Dis'
                item_predict.append(mentionType)
                predictions.append(tuple(item_predict))
                g.write('\n')
            g.write('\n')           
        g.close()

        dutil.copy_predictions_to_predictions_with_header(self.prediction_filename)

        return predictions


    def evaluate(self, predictions, groundTruths, *args,
                      **kwargs):  # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
        """
        Calculates evaluation metrics on chosen benchmark dataset [Precision,Recall,F1, or others...]

        NOTE:
            Empty arrays or None for preconditions and groundTruths will assume a 'predictions.txt' file
            exists, and that file will be used for evaluation data! This is preferred
        Args:
            predictions: [tuple,...], list of tuples [same format as output from predict]. Or None.
            groundTruths: [tuple,...], list of tuples representing ground truth. Or None
                PRECONDITION: parallel arrays predictions and groundTruths. must be same length and
                    each element must correspond to the same item [i.e. token]. If this is not the case,
                    evaluation will not be accurate.
        Returns:
            metrics: tuple with (p,r,f1). Each element is float. (None,None,None) if metrics cannot be calculated to do div-by-zero error
        Raises:
            None
        """
        # pseudo-implementation
        # we have a set of predictions and a set of ground truth data.
        # calculate true positive, false positive, and false negative
        # calculate Precision = tp/(tp+fp)
        # calculate Recall = tp/(tp+fn)
        # calculate F1 using precision and recall
        assume_predictions_file = False
        if not predictions:  # predictions is empty or None
            assume_predictions_file = True
        if not groundTruths:  # groundTruths is empty or None
            assume_predictions_file = True

        truths = []
        preds = []
        if assume_predictions_file:
            #load predictions data
            filedata = np.genfromtxt(self.prediction_filename,delimiter=' ',dtype=str,comments=None)
            truths = filedata[:,1]
            preds = filedata[:,2]
        else:
            preds = np.asarray([item[-1] for item in predictions],dtype=str)
            truths = np.asarray([item[-1] for item in groundTruths],dtype=str)

        # build list of tuples with (idx_start,idx_end)
        truth_entities = set()
        startIdx = None
        endIdx = None
        is_entity = False
        for idx in range(len(truths)):
            if truths[idx]=='B-Dis':
                is_entity = True
                startIdx = idx
                endIdx = idx
                continue
            if is_entity:
                if truths[idx]=='B-Dis':  # corner case...B-Dis followed by B-Dis
                    truth_entities.add(tuple([startIdx,endIdx]))
                    startIdx = idx
                    endIdx = idx
                    #still keep is_entity
                    continue
                if truths[idx]=='I-Dis':
                    endIdx = idx
                    #still keep is_entity
                    continue
                if truths[idx]=='O':  # no longer an entity
                    truth_entities.add(tuple([startIdx,endIdx]))
                    startIdx = None
                    endIdx = None
                    is_entity = False
                    continue
        # build list of tuples with (idx_start,idx_end)
        pred_entities = set()
        startIdx = None
        endIdx = None
        is_entity = False
        for idx in range(len(preds)):
            if preds[idx]=='B-Dis':
                is_entity = True
                startIdx = idx
                endIdx = idx
                continue
            if is_entity:
                if preds[idx]=='B-Dis':  # corner case...B-Dis followed by B-Dis
                    pred_entities.add(tuple([startIdx,endIdx]))
                    startIdx = idx
                    endIdx = idx
                    #still keep is_entity
                    continue
                if preds[idx]=='I-Dis':
                    endIdx = idx
                    #still keep is_entity
                    continue
                if preds[idx]=='O':  # no longer an entity
                    pred_entities.add(tuple([startIdx,endIdx]))
                    startIdx = None
                    endIdx = None
                    is_entity = False
                    continue

        # get tp
        tp = float(len(truth_entities.intersection(pred_entities)))
        print 'tp', tp

        # get fp
        fp = float(len(pred_entities.difference(truth_entities)))
        print 'fp', fp

        # get fn
        fn = float(len(truth_entities.difference(pred_entities)))
        print 'fn', fn

        if tp == 0:
            if (fp == 0) or (fn == 0):
                print('Results are poor such that stats cannot be properly calculated. true_positives: %s, false_positives: %s, false_negatives: %s'%(tp,fp,fn))
                return (None,None,None)

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)

        print 'final evaluation scores: precision={}, recall={}, f1={}'.format(str(precision),str(recall),str(f1))


        return (precision, recall, f1)


    def save_model(self, file):
        """
        :param file: Where to save the model - Optional function
        :return:
        """

        with open(file,'wb') as sm:
            cPickle.dump(self.trained_model, sm, protocol=cPickle.HIGHEST_PROTOCOL)

        return
    

    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """

        with open(file,'rb') as sm:
            self.trained_model = cPickle.load(sm)
            if self.trained_model == None:
                print('whoooooooop')

        return


    def unittest_main(self,inputFilePath):
        #instantiate a model!

        # test params:
        test_params = {'n_iter':501,'m_report':100,'save_checkpoint_files':False}
        drtrnn = disease_name_recognition_through_rnn(**test_params)

        # print('input file: %s'%inputFilePath)

        # print(type(myModel))

        # convert dataset to properformat used by training
        # 1] read_dataset()
        file_dict = {'train':{'data':inputFilePath},'dev':{},'test':{}}
        dataset_name = 'unittest'
        data = drtrnn.read_dataset(file_dict, dataset_name)  # data read, converted, and written to files in proper location expected by train
        # 2] intermediate step, generate *_tags files, *_words files, vocab file

        # train model
        #data = []
        data_train = data['train']  # test passing of actual data
        drtrnn.train(data_train)

        # predict using trained model
        data_test = data['test']  # test passing of actual data
        drtrnn.predict(data_test)

        outputPredictionsFile = 'predictions.txt'
        finalOutputFile = dutil.copy_predictions_to_predictions_with_header(raw_predictions_filename=outputPredictionsFile)

        # read from predictions file for evaluate
        evaluation_results = self.evaluate(None,None)

        # use direct data for evaluate
        # groundTruths = load_groundTruth_from_predictions(raw_predictions_filename=outputPredictionsFile)
        # evaluation_results = drtrnn.evaluate(predictions,groundTruths)

        print('%s'%str(evaluation_results))
    
        return finalOutputFile  # NOT FULLY IMPLEMENTED


# if __name__=='__main__':
#     finalOutputFileName = main('binary_data/text.txt')

# ----------------------------END IMPLEMENT PARENT CLASS ABASTRACTMETHODS------------------------------#


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