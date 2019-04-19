from graph.embedding.graph_embedding import GraphEmbedding
from datetime import datetime
import logging
import numpy as np
import os
from graph.embedding.analogy.dataset import TripletDataset, Vocab
from graph.embedding.analogy.optimizer import SGD, Adagrad
from graph.embedding.analogy.analogy_model import ANALOGYModel
from graph.embedding.analogy.evaluator import Evaluator
from graph.embedding.analogy.graph_util import TensorTypeGraph


np.random.seed(46)

DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H%M')))

INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\FB15k"


class ANALOGY(GraphEmbedding):
    def __init__(self, logger_path=None):
        # logger_path = kwargs.get("log_files_path", None)
        if logger_path is None:
            logger_path = DEFAULT_LOG_DIR
        if not os.path.exists(logger_path):
            os.mkdir(logger_path)
        # if not os.path.exists(args.log): OLD
        #    os.mkdir(args.log) OLD
        self.logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO)
        log_path = os.path.join(logger_path, 'log')
        file_handler = logging.FileHandler(log_path)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)
        self.logger.info(" Initializing ANALOGY ... ")

        self.n_entity = 0
        self.n_relation = 0
        self.train_dat = None
        self.valid_dat = None
        self.whole_graph = None
        self.model = None

    def read_dataset(self, file_names, *args, **kwargs):  # <--- implemented PER class
        """ Reads datasets and convert them to proper format for train or test.
            Returns data in proper format for train, validation and test.

        Args:
            file_names: list-like. List of files representing the dataset to read. Each element is str, representing
            filename [possibly with filepath]
            options: object to store any extra or implementation specific data
        
        Returns:
            data: data in proper [arbitrary] format for train, validation and test.
        Raises:
            None
        """

        self.logger.info('Input Files ...')
        for arg, val in sorted(file_names.items()):
            self.logger.info('{:>10} -----> {}'.format(arg, val))

        ent_vocab = Vocab.load(file_names["entities"])
        rel_vocab = Vocab.load(file_names["relations"])
        # rel_vocab = Vocab.load(args.rel) OLD
        self.n_entity, self.n_relation = len(ent_vocab), len(rel_vocab)

        # preparing data
        self.logger.info('preparing data...')
        # train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab) OLD
        # valid_dat = TripletDataset.load(args.valid, ent_vocab, rel_vocab) if args.valid else None OLD
        self.train_dat = TripletDataset.load(file_names["train"], ent_vocab, rel_vocab)
        self.valid_dat = TripletDataset.load(file_names["valid"], ent_vocab, rel_vocab) \
            if "valid" in file_names else None

        if "whole" in file_names:
            self.logger.info('loading whole graph...')
            self.whole_graph = TensorTypeGraph.load_from_raw(file_names["whole"], ent_vocab, rel_vocab)
        else:
            self.whole_graph = None

        # <--- implemented PER class

# ##############################################################################################################
# #### ERROR CHECK THAT FILES ARE LOADED ####
    def learn_embeddings(self, data, *args, **kwargs):
        """
        Learns embeddings with data, build model and train the model

        Args:
            data: iterable of arbitrary format. represents the data instances and features and labels you need to train your model.
            Note: formal subject to OPEN ITEM mentioned in read_dataset!
            options: object to store any extra or implementation specific data
        Returns:
            ret: None. Trained model stored internally to class instance state.

        Raises:
            None
        """

        self.logger.info('Arguments...')
        for arg, val in sorted(data.items()):
            self.logger.info('{:>10} -----> {}'.format(arg, val))

        if 'optimizer' in data:
            opt_method = data['optimizer']
        else:
            opt_method = data['optimizer'] = 'adagrad'

        # opt_method = data['optimizer'] if 'optimizer' in data else 'adagrad'
        learning_rate = data['lr'] if 'lr' in data else 0.05
        if opt_method == 'sgd':
            opt = SGD(learning_rate)
        elif opt_method == 'adagrad':
            opt = Adagrad(learning_rate)
        else:
            raise NotImplementedError

        l2_regularization = data['l2_reg'] if 'l2_reg' in data else 0.001
        if l2_regularization > 0:
            opt.set_l2_reg(l2_regularization)

        gradient_clipping = data['gradclip'] if 'gradclip' in data else 0.05
        if gradient_clipping > 0:
            opt.set_gradclip(gradient_clipping)

        number_dimensions = data['dim'] if 'dim' in data else 200
        margin = data['margin'] if 'margin' in data else 1
        cp_ratio = data['dim'] if 'dim' in data else .5

        # CHECK MODE AND ERROR ############################################################################
        mode = data['mode'] if 'mode' in data else 'single'

        self.model = ANALOGYModel(n_entity=self.n_entity,
                                  n_relation=self.n_relation,
                                  margin=margin,
                                  dim=number_dimensions,
                                  cp_ratio=cp_ratio,
                                  mode=mode)
        # CHECK MODE AND ERROR ############################################################################
        metric = data['metric'] if 'metric' in data else 'mrr'

        # CHECK MODE AND ERROR ############################################################################
        nbest = data['nbest'] if 'nbest' in data else 10 # only for hits@ metric

        filtered = data['filtered'] if 'filtered' in data else False

        evaluator = Evaluator(metric, nbest, filtered, self.whole_graph) \
            if self.valid_dat else None

        batch = data['batch'] if 'batch' in data else 128
        # batch
        save_step = data['save_step'] if 'save_step' in data else 30
        # save_step
        epoch = data['epoch'] if 'epoch' in data else 500
        # epoch
        negative = data['negative'] if 'negative' in data else 5
        # negative

        if filtered and self.valid_dat:
            evaluator.prepare_valid(self.valid_dat)
        if args.mode == 'pairwise':
            trainer = PairwiseTrainer(model=self.model, opt=opt, save_step=args.save_step,
                                      batchsize=args.batch, logger=logger,
                                      evaluator=evaluator, valid_dat=valid_dat,
                                      n_negative=args.negative, epoch=args.epoch,
                                      model_dir=args.log)
        elif args.mode == 'single':
            trainer = SingleTrainer(model=model, opt=opt, save_step=args.save_step,
                                    batchsize=args.batch, logger=logger,
                                    evaluator=evaluator, valid_dat=valid_dat,
                                    n_negative=args.negative, epoch=args.epoch,
                                    model_dir=args.log)
        else:
            raise NotImplementedError

    # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
    def evaluate(self, data, *args, **kwargs):
        """
        Predicts the embeddings with test data and calculates evaluation metrics on chosen benchmark dataset

        Args:
            data: data used to test the model, may need further process
            options: object to store any extra or implementation specific data

        Returns:
            metrics: cosine similarity, MMR or Hits

        Raises:
            None
        """

        results = {}
        return results
        print("evaluate")

    def save_model(self, file):
        """ saves model to file
        :param file: Where to save the model - Optional function
        :return:
        """
        print("save model")

    def load_model(self, file):
        """ loads model from file
        :param file: From where to load the model - Optional function
        :return:
        """
        print("load model")

"""
# Sample workflow:

inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']

myModel = myClass(ditk.Graph_Embedding)  # instatiate the class

data = myModel.read_dataset(inputFiles)  # read in a dataset for training

myModel.learn_embeddings(data)  # builds and trains the model and stores model state in object properties or similar

results = myModel.evaluate(data)  # calculate evaluation results

print(results)

"""

