from __future__ import print_function
import numpy as np
from numpy import argsort
from collections import defaultdict as ddict
import pickle
import timeit
import logging
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

from graph.embedding.HolE.skge import sample
from graph.embedding.HolE.skge.util import to_tensor

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EX-KG')
np.random.seed(42)


class FilteredRankingEval(object):

    def __init__(self, xs, true_triples, neval=-1): #xs: train data
        idx = ddict(list)
        tt = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        self.neval = neval
        self.sz = len(xs)
        for s, o, p in xs:
            idx[p].append((s, o))

        for s, o, p in true_triples:
            tt[p]['os'][s].append(o)
            tt[p]['ss'][o].append(s)

        self.idx = dict(idx)
        self.tt = dict(tt)

        self.neval = {}
        for p, sos in self.idx.items():
            if neval == -1:
                self.neval[p] = -1
            else:
                self.neval[p] = np.int(np.ceil(neval * len(sos) / len(xs)))

    def positions(self, mdl):
        pos = {}
        fpos = {}
        first = {}
        ffirst = {}
        if hasattr(self, 'prepare_global'):
            self.prepare_global(mdl)

        for p, sos in self.idx.items(): #p:predicate, sos: (subject,object)
            ppos = {'head': [], 'tail': []}
            pfpos = {'head': [], 'tail': []}
            pfirst = {'head': [], 'tail': []}
            pffirst = {'head': [], 'tail': []}
            if hasattr(self, 'prepare'):
                self.prepare(mdl, p)

            for s, o in sos[:self.neval[p]]:

                scores_o = self.scores_o(mdl, s, p).flatten()
                sortidx_o = argsort(scores_o)[::-1]

                ppos['tail'].append(np.where(sortidx_o == o)[0][0] + 1)
                pfirst['tail'].append(sortidx_o[0]+1)

                rm_idx = self.tt[p]['os'][s]
                rm_idx = [i for i in rm_idx if i != o]
                scores_o[rm_idx] = -np.Inf
                sortidx_o = argsort(scores_o)[::-1]
                pfpos['tail'].append(np.where(sortidx_o == o)[0][0] + 1)
                pffirst['tail'].append(sortidx_o[0] + 1)

                scores_s = self.scores_s(mdl, o, p).flatten()
                sortidx_s = argsort(scores_s)[::-1]
                ppos['head'].append(np.where(sortidx_s == s)[0][0] + 1)
                pfirst['head'].append(sortidx_s[0] + 1)

                rm_idx = self.tt[p]['ss'][o]
                rm_idx = [i for i in rm_idx if i != s]
                scores_s[rm_idx] = -np.Inf
                sortidx_s = argsort(scores_s)[::-1]
                pfpos['head'].append(np.where(sortidx_s == s)[0][0] + 1)
                pffirst['head'].append(sortidx_o[0] + 1)
            pos[p] = ppos
            fpos[p] = pfpos
            first[p] = pfirst
            ffirst[p] = pffirst

        return pos, fpos,first,ffirst





class Experiment(object):
    def __init__(self):
        self.margin = 0.2
        self.init = 'nunif'
        self.lr = 0.1
        self.me = 5
        self.ne = 1
        self.nb = 100
        self.fout = ''
        self.input_file =''
        self.test_all = 5
        self.no_pairwise = False
        self.mode = 'rank'
        self.sampler = 'random-mode'
        self.rparam = 0.0
        self.afs = 'sigmoid'
        self.ncomp = 300
        self.load = False
        self.eval = 0


        '''
        self.parser = argparse.ArgumentParser(prog='Knowledge Graph experiment', conflict_handler='resolve')
        self.parser.add_argument('--margin', type=float, help='Margin for loss function')
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate')
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs')
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches')
        self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)
        self.parser.add_argument('--fin', type=str, help='Path to input data', default=None)
        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=10)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)
        self.parser.add_argument('--mode', type=str, default='rank')
        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.parser.add_argument('--rparam', type=float, help='Regularization for W', default=0)
        self.parser.add_argument('--afs', type=str, default='sigmoid', help='Activation function')
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components',default=150)
        '''
        self.neval = -1
        self.best_valid_score = 1000000
        self.exectimes = []


    def run(self):
        #self.args = self.parser.parse_args()
        if self.mode == 'rank':
            self.callback = self.ranking_callback
        else:
            raise ValueError('Unknown experiment mode (%s)' % self.mode)

    def ranking_callback(self, trn, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - trn.epoch_start
        self.exectimes.append(elapsed)
        if self.no_pairwise:
            log.info("[%3d] time = %ds, loss = %f" % (trn.epoch, elapsed, trn.loss))
        else:
            log.info("[%3d] time = %ds, violations = %d" % (trn.epoch, elapsed, trn.nviolations))

        # if we improved the validation error, store model and calc test error

        if (trn.epoch % self.test_all == 0) or with_eval:

            pos_v, fpos_v, first_v, ffirst_v = self.ev_valid.positions(trn.model)
            fmrr_valid,fmean_pos_valid, fhits_valid = ranking_scores(pos_v, fpos_v, trn.epoch, 'VALID')

            log.debug("FMRR valid = %f, best = %f" % (fmrr_valid, self.best_valid_score))
            if fmrr_valid < self.best_valid_score:
                self.best_valid_score = fmrr_valid

        return True

def ranking_hits(pos, fpos, epoch):

    fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
    ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]
    hits_1= _print_pos_hits(
        np.array(fhpos + ftpos),1)
    hits_3 = _print_pos_hits(
        np.array(fhpos + ftpos), 3)

    return hits_1,hits_3

def ranking_scores(pos, fpos, epoch, txt):

    hpos = [p for k in pos.keys() for p in pos[k]['head']]
    tpos = [p for k in pos.keys() for p in pos[k]['tail']]
    fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
    ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]
    fmrr,fmean_pos, fhits= _print_pos(
        np.array(hpos + tpos),
        np.array(fhpos + ftpos),
        epoch, txt)

    return fmrr,fmean_pos, fhits

def _print_pos_hits(fpos, hits):

    fmrr, fmean_pos, fhits = compute_scores(fpos,hits)

    return fhits

def _print_pos(pos, fpos, epoch, txt):
    mrr, mean_pos, hits = compute_scores(pos,10)
    fmrr, fmean_pos, fhits = compute_scores(fpos,10)

    return fmrr,fmean_pos, fhits


def compute_scores(pos, hits):

    mrr = np.mean(pos)
    mean_pos = np.mean(pos)
    hits = np.mean(pos <= hits).sum() * 100

    return mrr, mean_pos, hits

def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)

def pre_process_data(fin):
    yago_data = {}
    relation2id = {}
    r = 0
    e = 0
    entity2id = {}
    relations = []
    entities = []
    e = 0
    r = 0
    c1 = open(fin[0],'r')
    lines = c1.readlines()
    c1.close()
    tuples = []
    for line in lines:
        line = line.replace('\n','')
        line = line.split()
        if len(line) >=3:
            if line[0] not in entities:
                entity2id[line[0]] = e
                e+=1
                entities.append(line[0])
            if line[1] not in relations:
                relation2id[line[1]] =r
                r += 1
                relations.append(line[1])
            if line[2] not in entities:
                entity2id[line[2]] = e
                e += 1
                entities.append(line[2])

            tuples.append((entity2id[line[0]], entity2id[line[2]],relation2id[line[1]]))
    yago_data['train_subs'] = tuples

    c1 = open(fin[1], 'r')
    lines = c1.readlines()
    c1.close()
    tuples = []
    for line in lines:
        line = line.replace('\n', '')
        line = line.split()
        if len(line) >= 3:
            if line[0] not in entities:
                entity2id[line[0]] = e
                e += 1
                entities.append(line[0])
            if line[1] not in relations:
                relation2id[line[1]] = r
                r += 1
                relations.append(line[1])
            if line[2] not in entities:
                entity2id[line[2]] = e
                e += 1
                entities.append(line[2])

            tuples.append((entity2id[line[0]], entity2id[line[2]], relation2id[line[1]]))
    yago_data['valid_subs'] = tuples

    c1 = open(fin[2], 'r')
    lines = c1.readlines()
    c1.close()
    tuples = []
    for line in lines:
        line = line.replace('\n', '')
        line = line.split()
        if len(line) >= 3:
            if line[0] not in entities:
                entity2id[line[0]] = e
                e += 1
                entities.append(line[0])
            if line[1] not in relations:
                relation2id[line[1]] = r
                r += 1
                relations.append(line[1])
            if line[2] not in entities:
                entity2id[line[2]] = e
                e += 1
                entities.append(line[2])

            tuples.append((entity2id[line[0]], entity2id[line[2]], relation2id[line[1]]))
    yago_data['test_subs'] = tuples

    yago_data['relations'] = relations
    yago_data['entities'] = entities


    return yago_data








