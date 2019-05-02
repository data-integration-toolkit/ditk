#import graph_completion
from optimize import *
from diagnostics import *
from configs import *
import argparse
from data import *
import copy
import util
import  numpy as np



class GraphTraverse():
    # The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group.

    def __init__(self):
        self.max_negative_samples_train = None
        self.positive_branch_factor = None
        self.type_matching_negs = None
        self.path_model = None
        self.hidden_dim = None
        self.init_scale = None
        self.glove_path = None
        self.wvec_dim = None
        self.report_wait = None
        self.dataset_path = None
        self.params_path = None
        self.max_steps_single = None
        self.step_size_single = None
        self.max_negative_samples_eval = None
        self.l2_reg = None
        self.batch_size = None
        self.max_steps_path = None
        self.step_size_path = None
        self.dset = None
        self.init_params = None
        self.trainer0 = None
        self.trainer = None
        self.params_comp = None
        self.params_single = None
        self.one_hop_only=None

    def read_dataset(self, fileName, options):
        """
        Reads dataset from the benchmark files.
        dataset will be of the "subject-relation-object". Subject and Object are the entities.
        Converts the data into numerical format.
        Preprocesses the data to further add some negative examples.
        Then it uses bilinear vector space model to form a vector space of entities and relations
        Splits the data into train, dev and test sets.

        """
        self.dataset_path=fileName
        dict=None
        if (options == "FreeBase"):
            dict=freebase_experiment
        else:
            dict=wordnet_experiment

        self.max_negative_samples_train = dict['max_negative_samples_train']
        self.positive_branch_factor = dict['positive_branch_factor']
        self.type_matching_negs = dict['type_matching_negs']
        self.path_model = dict['path_model']
        self.hidden_dim = dict['hidden_dim']
        self.init_scale = dict['init_scale']
        self.glove_path = fileName[2]
        self.wvec_dim = dict['wvec_dim']
        self.report_wait = dict['report_wait']
        self.dataset_path=fileName[0]
        self.params_path = fileName[1]
        self.max_steps_single = dict['max_steps_single']
        self.step_size_single = dict['step_size_single']
        self.max_negative_samples_eval = dict['max_negative_samples_eval']
        self.l2_reg = dict['l2_reg']
        self.batch_size = dict['batch_size']
        self.max_steps_path = dict['max_steps_path']
        self.step_size_path = dict['step_size_path']
        self.dset = parse_dataset(self.dataset_path, dev_mode=False, maximum_examples=100)
        warm_start = self.params_path is not None

        if warm_start:
            print('loading warm start params...')
            self.init_params = load_params(self.params_path, self.path_model)
        else:
            self.init_params = None

    def build_trainer(self,train, test, max_steps, step_size, init_params=None):
        # negative triple generator for training
        triples = [(q.s, str(q.r[0]), q.t) for q in train if len(q.r) == 1]
        print("came here")
        train_graph = Graph(triples)
        train_neg_gen = NegativeGenerator(train_graph, self.max_negative_samples_train,
                                          self.positive_branch_factor, self.type_matching_negs
                                          )

        # specify the objective to maximize
        print("came here 2")
        objective = CompositionalModel(train_neg_gen, path_model=self.path_model,
                                       objective='margin')

        # initialize params if not already initialized
        if init_params is None:
            init_params = objective.init_params(
                self.dset.entity_list, self.dset.relations_list, self.wvec_dim, model=self.path_model,
                hidden_dim=self.hidden_dim, init_scale=self.init_scale, glove_path=self.glove_path)

        save_wait = 1000  # save parameters after this many steps
        eval_samples = 200  # number of examples to compute objective on

        # define Observers
        observers = [NormObserver(self.report_wait), SpeedObserver(self.report_wait),
                     ObjectiveObserver(eval_samples, self.report_wait)]

        # this Observer computes the mean rank on each split
        rank_observer = RankObserver({'train': train, 'test': test},
                                     self.dset.full_graph, eval_samples,
                                     self.max_negative_samples_eval, self.report_wait,
                                     type_matching_negs=True)
        observers.append(rank_observer)

        # define Controllers
        controllers = [BasicController(self.report_wait, save_wait, max_steps),
                       DeltaClipper(), AdaGrad(), UnitNorm()]

        trainer = OnlineMaximizer(
            train, test, objective, l2_reg=self.l2_reg, approx_reg=True,
            batch_size=self.batch_size, step_size=step_size, init_params=init_params,
            controllers=controllers, observers=observers)

        return trainer

    def train(self):
        """
         Using the vector space obtained, Compositional training using Bilinear-diag model is done.
         There are changes in the optimization function of the model, changes are according to the paper.
         Model is made more robust using the dev data.
        """
        # train the model on single edges

        self.one_hop_only = lambda queries: [q for q in queries if len(q.r) == 1]
        one_hop_only=self.one_hop_only

        train=self.dset.train
        test=self.dset.test
        init_params=self.init_params

        step_size_single=self.step_size_single
        print(step_size_single)
        max_steps_single=self.max_steps_single
        print(max_steps_single)
        self.trainer0 = self.build_trainer(one_hop_only(train),one_hop_only(test),max_steps_single, step_size_single,init_params)
        self.params0 = self.trainer0.maximize()
        self.params_single = copy.deepcopy(self.params0)


        # train the model on all edges, with warm start from single-edge model
        self.trainer = self.build_trainer(self.dset.train, self.dset.test,self.max_steps_path,self.step_size_path, self.params0)
        self.params_comp = self.trainer.maximize()



    def predict(self, data, options={}):
        """
         from the test data we predict object on the basis of subject- relation.
         for every subject-relation vector a set of entities(objects) are predicted.
         This set of entities are ordered on the basis of likelihood.
        """
        model=self.trainer.objective
        score= model.predict(self.params_comp,data)
        if(score>0):
            return True
        else:
            return False



    def report(self,queries, model, neg_gen, params):
        scores = lambda query: model.predict(params, query).ravel()

        def compute_quantile(query):
            s, r, t = query.s, query.r, query.t
            negatives = neg_gen(query, 't')
            pos_query = PathQuery(s, r, t)
            neg_query = PathQuery(s, r, negatives)

            # don't score queries with no negatives
            if len(negatives) == 0:
                query.quantile = np.nan
            else:
                query.quantile = util.average_quantile(scores(pos_query), scores(neg_query))

            query.num_candidates = len(negatives) + 1

        for query in util.verboserate(queries):
            compute_quantile(query)

        # filter out NaNs
        queries = [q for q in queries if not np.isnan(q.quantile)]
        mean_quantile = np.mean([q.quantile for q in queries])
        hits_at_10 = np.mean(
            [1.0 if util.rank_from_quantile(q.quantile, q.num_candidates) <= 10 else 0.0 for q in queries])

        print('mean_quantile:', mean_quantile)
        print('h10', hits_at_10)

        return mean_quantile, hits_at_10

    def evaluate(self):
        """
        Evaluates the predicted results using evaluation metrics.
        """
        # used for all evaluations
        neg_gen = NegativeGenerator(self.dset.full_graph, float('inf'), type_matching_negs=True)

        print('path query evaluation')

        print('--Single-edge trained model--')
        mq, h10 = self.report(self.dset.test, self.trainer0.objective, neg_gen, self.params_single)
        util.metadata(('path_queries', 'SINGLE', 'mq'), mq)
        util.metadata(('path_queries', 'SINGLE', 'h10'), h10)
        print

        print('--Compositional trained model--')
        mq, h10 = self.report(self.dset.test, self.trainer.objective, neg_gen, self.params_comp)
        util.metadata(('path_queries', 'COMP', 'mq'), mq)
        util.metadata(('path_queries', 'COMP', 'h10'), h10)
        print

        print('single edge evaluation')
        print('--Single-edge trained model--')

        mq, h10 = self.report(self.one_hop_only(self.dset.test), self.trainer0.objective, neg_gen, self.params_single)
        util.metadata(('single_edges', 'SINGLE', 'mq'), mq)
        util.metadata(('single_edges', 'SINGLE', 'h10'), h10)
        print

        print('--Compositional trained model--')
        mq, h10 = self.report(self.one_hop_only(self.dset.test), self.trainer.objective, neg_gen, self.params_comp)
        util.metadata(('single_edges', 'COMP', 'mq'), mq)
        util.metadata(('single_edges', 'COMP', 'h10'), h10)
