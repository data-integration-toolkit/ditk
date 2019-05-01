
#
# The parent class is located at: https://github.com/sandiexie-USC/spring19_csci548_graphembedding
#
from __future__ import print_function
from graph.embedding.HolE.base import Experiment

import logging
from graph.embedding.HolE.skge import sample
from graph.embedding.HolE.skge import HolE
from graph.embedding.HolE.skge.base import PairwiseStochasticTrainer
from graph.embedding.HolE.base import cosin_distance
import numpy as np
from graph.embedding.HolE.base import FilteredRankingEval,ranking_scores,ranking_hits,pre_process_data
from graph.embedding.HolE.skge.util import ccorr
import pickle
import pandas as pd
logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger('EX-KG')


class HolEEval(FilteredRankingEval):
    def prepare(self, mdl, p):
        self.ER = ccorr(mdl.R[p], mdl.E)

    def scores_o(self, mdl, s, p):
        return np.dot(self.ER, mdl.E[s])

    def scores_s(self, mdl, o, p):
        return np.dot(mdl.E, self.ER[o])

class HolE_Embedding(Experiment):

    def _init_(self):
        # declare model
        super(HolE_Embedding, self).__init__()


    def read_dataset(self, fileName='', *arg, **kwarg):
        # read the files and get data for training, validaton and testing
        # inputs:
        #   fileNames: list of file names, which could be file pathes
        #   options: other parameters
        # returns:
        #   data read from datasets

        # implementation:
        #   read data from dataset
        #   convert data format and write into new files
        #   read data from new files


        self.evaluator = HolEEval
        self.run()
        self.data = pre_process_data(self.input_file)

        return self.data['train_subs'],self.data['valid_subs'],self.data['test_subs'], self.data['entities'], self.data['relations']

    def learn_embeddings(self, fileName='', *arg, **kwarg):
        # Learns embeddings with data, build model and train the model
        # inputs:
        #   data: data used to train the model and generate negative examples
        #   options: other parameters
        #
        # returns:
        #   None, model and embeddings are stored internally

        # implementation:
        #   generate negative samples
        #   build model
        #   train model (updating model with loss function)

        self.data = {}
        self.data['train_subs'] = self.train
        self.data['valid_subs'] = self.valid
        self.data['test_subs'] = self.test
        self.data['entities'] = self.entities
        self.data['relations'] = self.relations

        N = len(self.data['entities'])
        M = len(self.data['relations'])
        sz = (N,N,M)

        true_triples = self.data['train_subs'] + self.data['test_subs'] + self.data['valid_subs']

        self.ev_test = self.evaluator(self.data['test_subs'], true_triples, self.neval)

        self.ev_valid = self.evaluator(self.data['valid_subs'], true_triples, self.neval)

        xs = self.data['train_subs']
        ys = np.ones(len(xs))


        sampler = sample.RandomModeSampler(self.ne, [0, 1], xs, sz)

        model = HolE(
            sz,
            self.ncomp,
            rparam=self.rparam,
            af=self.afs,
            init=self.init
        )
        trainer = PairwiseStochasticTrainer(
            model,
            nbatches=self.nb,
            max_epochs=self.me,
            post_epoch=[self.callback],
            learning_rate=self.lr,
            margin=self.margin,
            samplef=sampler.sample
        )
            # create sampling objects

        trn = trainer
        log.info("Fitting model %s with trainer %s" % (
            trn.model.__class__.__name__,
            trn.__class__.__name__)
                 )
        trn.fit(xs, ys)
        self.model = trn.model

        self.trn = trn
        self.callback(self.trn, with_eval=True)

        self.save_model()
        model = trn.model
        E = list(model.E)
        R = list(model.R)
        return [E + R, model, self.ev_test]

    def save_model(self,path=''):
        with open("model.bin", 'wb') as fout:
            pickle.dump(self.trn.model, fout,protocol=2)

    def load_model(self,path = ''):
        c1 = open("model.bin", "rb")
        self.model = pickle.load(c1)
        self.load = True


    def evaluate(self, fileName='', *arg, **kwarg):
        # Predicts the embeddings with test data and calculates evaluation metrics on chosen benchmark dataset
        # inputs:
        #   data: data used to predict result and evaluate
        #   options: other parameters
        # returns:
        #   evaluation results

        # implementations:
        #   model receive data and predict results
        #   evaluate
        #st = self.callback(self.trn, with_eval=True)
        epoch = self.me
        pos_t, fpos_t, first_t, ffirst_t = self.ev_test.positions(self.model)

        ranking_scores(pos_t, fpos_t,epoch,'TEST')

        fmrr_test, fmean_pos_test, fhits_test = ranking_scores(pos_t, fpos_t, epoch, 'TEST')
        fhits_1, fhits_3 = ranking_hits(pos_t, fpos_t, epoch)

        predicted = {}
        predicted['head'] = {}
        predicted['tail'] = {}

        result = {}
        result['cosine similarity'] = {}
        result['MRR'] = 0.0
        result['Hits'] = {}

        data = self.test
        target = {}
        target['head'] = {}
        target['tail'] = {}

        for l in data:
            if l[2] not in target['head']:
                target['head'][l[2]] = []
                target['tail'][l[2]] = []
            target['head'][l[2]].append(l[0])
            target['tail'][l[2]].append(l[1])

        fpos_head = {}
        fpos_tail = {}
        ffirst_head = {}
        ffirst_tail = {}

        for k in fpos_t:
            fpos_head[k] = fpos_t[k]['head']
            fpos_tail[k] = fpos_t[k]['tail']
        for k in ffirst_t:
            ffirst_head[k] =ffirst_t[k]['head']
            ffirst_tail[k] = ffirst_t[k]['tail']

        for l in data:
            if l[2] not in predicted['head']:
                predicted['head'][l[2]] = []
                predicted['tail'][l[2]] = []
            predicted['head'][l[2]] = [i - 1 for i in ffirst_head[l[2]]]
            predicted['tail'][l[2]] = [i - 1 for i in ffirst_tail[l[2]]]

        predicted_triples = {}
        predicted_triples['head'] = []
        predicted_triples['tail'] = []

        for d in data:
            for o in predicted['tail'][d[2]]:
                predicted_triples['head'].append((d[0], o, d[2]))
            for s in predicted['head'][d[2]]:
                predicted_triples['tail'].append((s, d[1], d[2]))

        for k in fpos_head:
            result['cosine similarity'][k] = {}
            result['cosine similarity'][k]['head'] = []
            for i in range(len(fpos_head[k])):
                result['cosine similarity'][k]['head'].append(cosin_distance(self.model.E[target['head'][k][i]], self.model.E[ffirst_head[k][i]-1]))

        for k in fpos_tail:
            result['cosine similarity'][k]['tail'] = []
            for i in range(len(fpos_tail[k])):
                result['cosine similarity'][k]['tail'].append(cosin_distance(self.model.E[target['tail'][k][i]], self.model.E[ffirst_tail[k][i]-1]))
        cos = 0
        count = 0
        for k in fpos_tail:
            result['cosine similarity'][k]['head'] = np.average(result['cosine similarity'][k]['head'])
            result['cosine similarity'][k]['tail'] = np.average(result['cosine similarity'][k]['tail'])
            if not pd.isnull(result['cosine similarity'][k]['head']) and not pd.isnull(result['cosine similarity'][k]['tail']):
                result['cosine similarity'][k] = (result['cosine similarity'][k]['head'] + result['cosine similarity'][k]['tail'])/2
                count+=1
                cos += result['cosine similarity'][k]

        result['cosine similarity'] = cos/count

        result['MR'] = fmrr_test

        result['Hits']['1'] = fhits_1
        result['Hits']['3'] =fhits_3
        result['Hits']['10'] = fhits_test

        return result
    def return_parameter(self):
        self.train, self.valid, self.test, self.entities, self.relations = self.read_dataset()
        [embedding_vector, model, ev_test] = self.learn_embeddings()
        return [model, ev_test]


    def main(self,fileName):

        self.input_file = fileName
        prefix  =''
        if fileName[0].rfind('/') >=0:
            prefix = fileName[0][:fileName[0].rfind('/')] + '/'
        self.output_file = prefix + 'output.txt'

        self.train, self.valid, self.test, self.entities, self.relations = self.read_dataset()

        [embedding_vector, model, ev_test] = self.learn_embeddings()

        c1 = open(self.output_file, "w")

        for i in range(len(model.E)):
            c1.write(str(self.entities[i]) + " " + str(np.array(model.E[i])) + '\n')
        for i in range(len(model.R)):
            c1.write(str(self.relations[i]) + " " + str(np.array(model.R[i])) + '\n')
        c1.close()

        result = self.evaluate(fileName, model,ev_test)

        print("MR: " + str(result['MR']) + '\n')
        print("Hits@1: " + str(result['Hits']['1']) + '\n')
        print("Hits@3: " + str(result['Hits']['3']) + '\n')
        print("Hits@10: " + str(result['Hits']['10']) + '\n')

        return self.output_file
