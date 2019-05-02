from graph.embedding.graph_embedding import GraphEmbedding

from datetime import datetime
import logging
import numpy as np
import os
import dill
from graph.embedding.complex.dataset import TripletDataset, Vocab
from graph.embedding.complex.optimizer import SGD, Adagrad
from graph.embedding.complex.complex_model import ComplExModel
from graph.embedding.complex.evaluator import Evaluator
from graph.embedding.complex.graph_util import TensorTypeGraph
from graph.embedding.complex.trainer import PairwiseTrainer, SingleTrainer

np.random.seed(46)

DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H%M')))


class ComplEx(GraphEmbedding):
    def __init__(self):
        self.n_entity = 0
        self.n_relation = 0
        self.train_dat = None
        self.valid_dat = None
        self.whole_graph = None
        self.model = None
        self.ent_vocab = None
        self.rel_vocab = None
        self.logger = None
        self.log_path = None

    def prepare_logger(self, logger_path):
        # logger_path = kwargs.get("log_files_path", None)
        if logger_path is None:
            logger_path = DEFAULT_LOG_DIR
        self.log_path = logger_path

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
        # self.logger.info(" Initializing ANALOGY ... ")

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
        logger_path = kwargs.get("logger_path", None)
        self.prepare_logger(logger_path)

        self.logger.info('Input Files ...')
        for arg, val in sorted(file_names.items()):
            self.logger.info('{:>10} -----> {}'.format(arg, val))

        self.ent_vocab = Vocab.load(file_names["entities"])
        self.rel_vocab = Vocab.load(file_names["relations"])

        # rel_vocab = Vocab.load(args.rel) OLD
        self.n_entity, self.n_relation = len(self.ent_vocab), len(self.rel_vocab)

        # preparing data
        self.logger.info('Preparing data...')
        # train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab) OLD
        # valid_dat = TripletDataset.load(args.valid, ent_vocab, rel_vocab) if args.valid else None OLD
        self.train_dat = TripletDataset.load(file_names["train"], self.ent_vocab, self.rel_vocab)
        self.valid_dat = TripletDataset.load(file_names["valid"], self.ent_vocab, self.rel_vocab) \
            if "valid" in file_names else None

        if "whole" in file_names:
            self.logger.info('   Loading whole graph...')
            self.whole_graph = TensorTypeGraph.load_from_raw(file_names["whole"], self.ent_vocab, self.rel_vocab)
        else:
            self.whole_graph = None

        self.logger.info('   Done loading data...')

        # <--- implemented PER class

    def input_var_utility(self, data_dict, argument, default_value):
        if argument in data_dict:
            ret = data_dict[argument]
        else:
            ret = data_dict[argument] = default_value
        return ret

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
        if self.train_dat is None:
            print("No Data Loaded - Use read_dataset method before attempting to train")
            return

        self.logger.info('Learning Embeddings...')

        # set up arguments and defaults
        opt_method = self.input_var_utility(data, 'opt', 'adagrad')
        learning_rate = self.input_var_utility(data, 'lr', 0.05)
        l2_regularization = self.input_var_utility(data, 'l2_reg', 0.001)
        gradient_clipping = self.input_var_utility(data, 'gradclip', 0.05)
        number_dimensions = self.input_var_utility(data, 'dim', 200)
        margin = self.input_var_utility(data, 'margin', 1)
        cp_ratio = self.input_var_utility(data, 'cp_ratio', 0.5)
        # CHECK MODE AND ERROR ############################################################################
        mode = self.input_var_utility(data, 'mode', 'single')
        metric = data['metric'] if 'metric' in data else 'mrr'
        # CHECK MODE AND ERROR ############################################################################
        nbest = self.input_var_utility(data, 'nbest', 10)
        filtered = self.input_var_utility(data, 'filtered', False)
        batch = self.input_var_utility(data, 'batch', 128)
        save_step = self.input_var_utility(data, 'save_step', 30)
        epoch = self.input_var_utility(data, 'epoch', 500)
        negative = self.input_var_utility(data, 'negative', 5)

        self.logger.info('Arguments...')
        for arg, val in sorted(data.items()):
            self.logger.info('{:>10} -----> {}'.format(arg, val))

        if opt_method == 'sgd':
            opt = SGD(learning_rate)
        elif opt_method == 'adagrad':
            opt = Adagrad(learning_rate)
        else:
            raise NotImplementedError

        if l2_regularization > 0:
            opt.set_l2_reg(l2_regularization)

        if gradient_clipping > 0:
            opt.set_gradclip(gradient_clipping)

        self.model = ComplExModel(n_entity=self.n_entity,
                                  n_relation=self.n_relation,
                                  margin=margin,
                                  dim=number_dimensions,
                                  cp_ratio=cp_ratio,
                                  mode=mode)
        # CHECK MODE AND ERROR ############################################################################

        # My Add - Also store in model so can be pickled
        self.model.rels_to_save = self.rel_vocab
        self.model.ents_to_save = self.ent_vocab

        evaluator = Evaluator(metric, nbest, filtered, self.whole_graph) \
            if self.valid_dat else None

        if filtered and self.valid_dat:
            evaluator.prepare_valid(self.valid_dat)
        if mode == 'pairwise':
            trainer = PairwiseTrainer(model=self.model, opt=opt, save_step=save_step,
                                      batchsize=batch, logger=self.logger,
                                      evaluator=evaluator, valid_dat=self.valid_dat,
                                      n_negative=negative, epoch=epoch,
                                      model_dir=self.log_path)
        elif mode == 'single':
            trainer = SingleTrainer(model=self.model, opt=opt, save_step=save_step,
                                    batchsize=batch, logger=self.logger,
                                    evaluator=evaluator, valid_dat=self.valid_dat,
                                    n_negative=negative, epoch=epoch,
                                    model_dir=self.log_path)
        else:
            raise NotImplementedError

        trainer.fit(self.train_dat)

        self.logger.info('done all')

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
        # ################################################## CHECK IF HAVE MODEL

        suppress_output = kwargs.get("suppress_output", True)

        print("Running model evaluation")
        # ent_vocab = Vocab.load(data["entities"])
        # rel_vocab = Vocab.load(data["relations"])
        ent_vocab = self.model.ents_to_save
        rel_vocab = self.model.rels_to_save

        # preparing data
        test_dat = TripletDataset.load(data["test"], ent_vocab, rel_vocab)

        filtered = True if 'whole' in data else False

        if filtered:
            print('loading whole graph...')
            # from utils.graph import TensorTypeGraph
            self.whole_graph = TensorTypeGraph.load_from_raw(data['whole'], ent_vocab, rel_vocab)
        else:
            self.whole_graph = None

        evaluator = Evaluator('all', None, filtered, self.whole_graph)

        if filtered:
            evaluator.prepare_valid(test_dat)
        # model = Model.load_model(args.model)

        all_res = evaluator.run_all_matric(self.model, test_dat)
        if not suppress_output:
            for metric in sorted(all_res.keys()):
                print('{:20s}: {}'.format(metric, all_res[metric]))

        return all_res

    def save_model(self, file):
        """ saves model to file
        :param file: Where to save the model - Optional function
        :return:
        """
        # ################################################# CHECK IF MODEL PRESENT
        print("Saving model: " + file)
        with open(file, 'wb') as fw:
            dill.dump(self.model, fw)
        print("  Model saved:")

        # self.model.save_model(file)

    def load_model(self, file):
        """ loads model from file
        :param file: From where to load the model - Optional function
        :return:
        """
        print("Loading model: " + file)
        with open(file, 'rb') as f:
            self.model = dill.load(f)

        print("   Model loaded")

    def retrieve_entity_embeddings(self, words):
        # ################################################## CHECK IF HAVE MODEL
        entities = []
        for word in words:
            entities.append(self.model.ents_to_save.word2id[word])
        # entities = self.model.ents_to_save.word2id(words)
        # print(words)
        # print(entities)
        sub_re_emb, sub_im_emb = self.model.pick_ent(entities)
        return sub_re_emb, sub_im_emb
        # print(sub_emb)
        # print(sub_re_emb)
        # print(sub_im_emb)
        # for word in words:
        #     id = self.ent_vocab.word2id[word]
        # sub_re_emb, sub_im_emb, sub_emb = self.pick_ent(entities)

    def retrieve_relations_embeddings(self, words):
        # ################################################## CHECK IF HAVE MODEL
        relations = []
        for word in words:
            relations.append(self.model.rels_to_save.word2id[word])
        rel_re_emb, rel_im_emb = self.model.pick_ent(relations)
        return rel_re_emb, rel_im_emb

    def retrieve_scoring_matrix(self, sub_words, rels_words):
        # ################################################## CHECK IF HAVE MODEL
        subs = []
        rels = []
        for sw in sub_words:
            subs.append(self.model.ents_to_save.word2id[sw])
        for rw in rels_words:
            rels.append(self.model.rels_to_save.word2id[rw])
        sm = self.model.cal_scores(subs, rels)
        # return score matrix
        return sm
