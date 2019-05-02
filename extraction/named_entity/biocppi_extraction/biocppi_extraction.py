import abc
import sys
import pickle
from ner_2 import Ner  # <-- PARENT CLASS
from train import model_train
import biocppi_utils as butil
from six.moves import cPickle
import numpy as np

from model import BiLSTM

# -------------------End Helper for saving model------------------ #

"""
Abstraction for interface between DITKModel_NER and the code implementation of paper:
- Tung,T. and Kavuluru,R. An end-to-end deep learning architecture for
    extracting protein-protein interactions affected by genetic mutations.
    Database(2018) Vol. 2018: article ID bay092; doi:10.1093/database/bay092
"""

class biocppi_extraction(Ner):
    data_path_base = 'corpus_train/'  # CHANGE TO corpus_train?
    prediction_filename = 'predictions.txt'
    vocab_cache = data_path_base + 'word_vocab.ner.txt'
    labels = ['B-MISC', 'I-MISC', 'O']
    model_name = 'saved_model_autumn/'
    model_name_base = 'model'
    saved_model_base = data_path_base+model_name+model_name_base

# -------------------------------------init FOR MY CLASS------------------------------#
    def __init__(self,trained_model=[],embeddings_path='embeddings/PubMed-w2v.txt',other_datas={},**kwargs):
        self.trained_model = trained_model  # nothing to start. filled by train(). [] LIST OF MODELS
        self.cur_model = None  # index to hold a single model at a time

        self.embeddings_path = embeddings_path
        self.num_ensembles = 3  # number of full models to train! each model will train for a max of num_iterations
        self.optimizer = 'adam'  # optimizer to use, in 'default, rmsprop, adagrad, adam'. set to adam for now
        self.batch_size = 16  # batch size for training
        self.num_iterations = 5000  # number of iterations to run [an iteration is a parameter/weight update cycle from optimizer on singel batch] PER MODEL [i.e. single component of ensemble]
        self.num_it_per_ckpt = 100  # after this many iterations, save a checkpoint model file and consider it for best as this model [single ensemble]
        self.learning_rate = 'default'  # do not change
        self.embedding_factor = 1.0  # do not change
        self.decay_rate = 0.95  # do not change?
        self.keep_prob = 0.7  # do not change?
        self.num_cores = 4  # number of cores to exploit parallelism
        self.seed = 2  # random seed to use for pseudorandom initializations and reproducibility

        if 'num_ensembles' in kwargs:
            self.num_ensembles = kwargs['num_ensembles']
        if 'embeddings_path' in kwargs:
            self.embeddings_path = kwargs['embeddings_path']
        if 'optimizer' in kwargs:
            if kwargs['optimizer'] in ['default','rmsprop','adagrad','adam']:
                self.optimizer = kwargs['optimizer']
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        if 'num_iterations' in kwargs:
            self.num_iterations = kwargs['num_iterations']
        if 'num_it_per_ckpt' in kwargs:
            self.num_it_per_ckpt = kwargs['num_it_per_ckpt']
        if 'embedding_factor' in kwargs:
            self.embedding_factor = kwargs['embedding_factor']
        if 'decay_rate' in kwargs:
            self.decay_rate = kwargs['decay_rate']
        if 'keep_prob' in kwargs:
            self.keep_prob = kwargs['keep_prob']
        if 'num_cores' in kwargs:
            self.num_cores = kwargs['num_cores']
        if 'seed' in kwargs:
            self.seed = kwargs['seed']

        if self.num_it_per_ckpt > self.num_iterations:
            self.num_it_per_ckpt = int(self.num_iterations/2)  # requirement for num_it_per_ckpt < num_iterations

        self.other_datas = other_datas  # data structure for special model parameters/extra datas

# -------------------------------------END init FOR MY CLASS------------------------------#


# ----------------------------IMPLEMENT PARENT CLASS ABASTRACTMETHODS------------------------------#
    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        DEPRECATED. NO LONGER IN USE. USE butil.load_groundTruth_from_predictions()


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
        conversionFunctionMapper = {'CoNLL_2003':butil.convert_dataset_conll_to_train_format,
            'OntoNotes_5p0':butil.convert_dataset_ontoNotes_to_train_format,
            'CHEMDNER':butil.convert_dataset_chemdner_to_train_format,
            'ditk':butil.convert_ditk_to_train_format,
            'unittest':butil.convert_ditk_to_train_format}
            # 'ppim':self.convert_dataset_ppim_to_train_format, NOT YET IMPLEMENTED, but not required

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
            butil.write_drtrnn_format_to_file(data,self.data_path_base +'train.txt')  # NOT YET TESTED


        # temp1.model_train(N,M,n_f,n_hidden,model,viterbi,trainer,lr,batch_size,wl,tl,save_checkpoint_models=False)
        models = model_train(self.num_ensembles,self.data_path_base,self.model_name,self.embeddings_path,self.optimizer,
                    self.batch_size,self.num_iterations,self.num_it_per_ckpt,self.learning_rate,self.embedding_factor,
                    self.decay_rate,self.keep_prob,self.num_cores,self.seed)  # train the models! plural cuz this code does ensembles

        self.trained_model = models

        # if len(self.trained_model) < 1:
        #     print('Warning: No trained models to save.')
        #     return

        # for now, models are saved in train.py. no support of loading models at the moment
        # print('Saving trained models to dir: %s'%(self.data_path_base + self.model_name))
        # for i,model in enumerate(self.trained_model):
        #     print('Saving model %s'%i)
        #     model_save_filename = '%s_%s'%(self.saved_model_base,i)
        #     self.cur_model = model
        #     self.save_model(model_save_filename)

        # test loading of trained models only! NOT YET IMPLEMENTED
        # self.trained_model = []
        # for i in range(self.num_ensembles):
        #     model_save_filename = '%s_%s'%(self.saved_model_base,i)    
        #     m = self.load_model(model_save_filename)
        #     self.trained_model.append(m)
        # print(len(self.trained_model))

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

        # NOT YET IMPLEMENTED...loading of model...
        # if not self.trained_model:  # if trained_model is None or []
        #     self.trained_model = []
        #     for i in range(self.num_ensembles):
        #         try:
        #             load the model
        #         except:
        #             skip or something
        #         self.trained_model.append(model)
        # LEFT OFF HERE....

        if len(data) < 1:  # we were not passed data
            print('file based prediction not available at this time. Expect list of list for input data.')
            return None

        # self.trained_model = [1]  # TEMPORARY
        if not self.trained_model:
            print('No trained models to work with. Be sure to run train() before running this. Model loading not supported at this time')
            return None

        butil.write_drtrnn_format_to_file(data,self.data_path_base +'test.txt')
        sentences,sentence_labels = butil.load_dataset(self.data_path_base +'test.txt')

        predFile = open(self.prediction_filename,'w')
        predictions = []
        for ij, sentence in enumerate(sentences):
            # print >> sys.stderr, "Processing {}/{} {}".format(ij+1,len(pmids),pmid)
            flattened_len = sum([ len(x) for x in sentence ])
            proba_cumulative = np.zeros((flattened_len,len(self.labels)))

            for m in self.trained_model:
                proba_cumulative += m.predict_proba(sentence,batch_size=20)
            
            y_pred = np.argmax(proba_cumulative,axis=1)
            
            label_list = [ self.labels[tag_idx] for tag_idx in y_pred ]
            
            prediction = []
            extra_data = []
            for idx,x in enumerate(sentence):
                prediction.append(label_list[:len(x)])
                extra_data.append(sentence_labels[ij][idx])
                label_list = label_list[len(x):]
            
            # print '###' + pmid
            # print ''
            for idx_i,(line, tag) in enumerate(zip(sentence, prediction)):
                for idx_j,pair in enumerate(zip(line, tag)):
                    tokenword = pair[0]
                    predLabel = pair[1]
                    trueLabel = extra_data[idx_i][idx_j]
                    fullLine = [tokenword,trueLabel,predLabel]
                    # print ' '.join(fullLine)
                    outLine = ' '.join(fullLine)+'\n'
                    predFile.write(outLine)
                    predictions.append(tuple([None,None,tokenword,predLabel]))
                    # print ' '.join(pair) + ' ' + extra_data[idx_i][idx_j]  # token truth_label pred_label
                
                # print ''
                predFile.write('\n')
        predFile.close()

        butil.copy_predictions_to_predictions_with_header(self.prediction_filename)

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
            if truths[idx]=='I-MISC':
                is_entity = True
                startIdx = idx
                endIdx = idx
                continue
            if is_entity:
                if truths[idx]=='I-MISC':
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
            if preds[idx]=='I-MISC':
                is_entity = True
                startIdx = idx
                endIdx = idx
                continue
            if is_entity:
                if preds[idx]=='I-MISC':
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

        self.cur_model.save(file)

        return
    

    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """
        print('Model loading is unsupported at the moment. This may result in failure/crash due to tensorflow bug')
        with open(self.vocab_cache,'r') as f:
            word_vocab = pickle.load(f)

        m = BiLSTM(labels=self.labels,
                    word_vocab=word_vocab,
                    word_embeddings=None,
                    optimizer='default',
                    embedding_size=200, 
                    char_embedding_size=32,
                    lstm_dim=200,
                    num_cores=20,
                    embedding_factor=1.0,
                    learning_rate='default',
                    decay_rate=0.95,
                    dropout_keep=0.7)
        m.restore('./'+file)
        # self.trained_model.append(m)

        return m


    def unittest_main(self,inputFilePath):
        #instantiate a model!

        # test params:
        embeddings_path = '/home/kcrouch/USC/csci_548/project/groupedProject/bioNER_refactor/biocppi_extraction/embeddings/PubMed-w2v.txt'  # REMOVE FOR SUBMIT
        test_params = {'num_ensembles':2,'num_iterations':101,'num_it_per_ckpt':1000000}  # note, it num_it_per_ckpt > num_iterations then num_it_per_ckpt will be set to half of num_iterations
        test_params = {'num_ensembles':2,'num_iterations':101,'num_it_per_ckpt':1000000,'batch_size':2}  # for unittest, use small batch size
        biocppi = biocppi_extraction(embeddings_path=embeddings_path,**test_params)

        # convert dataset to properformat used by training
        # 1] read_dataset()

        # test unittest...good!
        dataset_name = 'unittest'
        file_dict = {'train':{'data':inputFilePath},'dev':{},'test':{}}
        
        data = biocppi.read_dataset(file_dict, dataset_name)  # data read, converted, and written to files in proper location expected by train
    
        # UNCOMMENT for submit!
        # train model
        #data = []  # implementation
        data_train = data['train']  # test passing actual data [empty also works]
        biocppi.train(data_train)
        print('DONE TRAIN')

        # predict using trained model
        data_test = data['test']
        predictions = biocppi.predict(data_test)  # test passing actual data [empty also works]
        print('DONE PREDICT')

        
        outputPredictionsFile = 'predictions.txt'
        finalOutputFile = butil.copy_predictions_to_predictions_with_header(raw_predictions_filename=outputPredictionsFile)
        
        # read from predictions file for evaluate
        evaluation_results = biocppi.evaluate(None,None)

        # use direct data for evaluate
        # groundTruths = load_groundTruth_from_predictions(raw_predictions_filename=outputPredictionsFile)
        # evaluation_results = biocppi.evaluate(predictions,groundTruths)

        print('%s'%str(evaluation_results))

        return finalOutputFile  # NOT FULLY IMPLEMENTED