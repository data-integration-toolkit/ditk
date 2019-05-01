#!/usr/bin/env python
import sys, os, random, pickle, json, codecs, fileinput
import numpy as np
from model import BiLSTM
import sklearn.metrics as skm
import argparse

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
parser.add_argument('--datapath', dest='datapath', type=str,
                    default='../fold_train', 
                    help='path to the datasets')
parser.add_argument('--embeddings', dest='embeddings_path', type=str,
                    default='../embeddings/PubMed-w2v.txt', 
                    help='path to the testing dataset')
parser.add_argument('--optimizer', dest='optimizer', type=str,
                    default='default', 
                    help='choose the optimizer: default, rmsprop, adagrad, adam.')
parser.add_argument('--batch-size', dest='batch_size', type=int, 
                    default=-1, help='number of instances in a minibatch')
parser.add_argument('--num-iterations', dest='num_iterations', type=int, 
                    default=10000, help='number of iterations')
parser.add_argument('--num-it-per-ckpt', dest='num_it_per_ckpt', type=int, 
                    default=100, help='number of iterations per checkpoint')
parser.add_argument('--learning-rate', dest='learning_rate', type=str,
                    default='default', help='learning rate')
parser.add_argument('--embedding-factor', dest='embedding_factor', type=float,
                    default=1.0, help='learning rate multiplier for embeddings')
parser.add_argument('--decay', dest='decay_rate', type=float,
                    default=0.95, help='exponential decay for learning rate')
parser.add_argument('--keep-prob', dest='keep_prob', type=float,
                    default=0.7, help='dropout keep rate')
parser.add_argument('--num-cores', dest='num_cores', type=int, 
                    default=20, help='seed for training')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=2, help='seed for training')

# num_ensembles = 10
num_ensembles = 7

def main(args):
    print >> sys.stderr, "Running BiLSTM model"
    print >> sys.stderr, args
    random.seed(args.seed)
    
    assert(os.path.isdir(args.datapath))
    vocab_cache = os.path.join(args.datapath,'word_vocab.ner.txt')

    print >> sys.stderr, "Loading vocab from {}..".format(vocab_cache)
    with open(vocab_cache,'r') as f:
        word_vocab = pickle.load(f)
    
    labels = ['B-MISC', 'I-MISC', 'O']

    lines = []
    lines.append('###999333111')  # bogus pmid to minimize changes elsewhere in code...
    for line in fileinput.input(args.files):
        lines.append(line)
        
    key = None
    tokens = {}
    tokens_data = {}
    curlist = []
    curdatalist = []
    pmids = []
    for l in lines:
        if l.startswith('###'):
            key = l[3:].rstrip()
            pmids.append(key)
            continue
        
        if key not in tokens:
            tokens[key] = []
            tokens_data[key] = []
        
        if l.strip() == '':
            if len(curlist) > 0:
                tokens[key].append(curlist)
                tokens_data[key].append(curdatalist)
            
            curlist = []
            curdatalist = []
        else:
            word = l.strip().strip().split(' ')[0].strip()
            word_info = l.strip().strip().split(' ')[1].strip()  # correct tag!
            # curlist.append(l.rstrip())
            curdatalist.append(word_info)
            curlist.append(word)
    
    models = []
    for j in range(num_ensembles):
        # Create the model, passing in relevant parameters
        m = BiLSTM(labels=labels,
                        word_vocab=word_vocab,
                        word_embeddings=None,
                            optimizer=args.optimizer,
                            embedding_size=200, 
                            char_embedding_size=32,
                            lstm_dim=200,
                            num_cores=args.num_cores,
                            embedding_factor=args.embedding_factor,
                            learning_rate=args.learning_rate,
                            decay_rate=args.decay_rate,
                            dropout_keep=args.keep_prob)
        
        save_path = '{}/saved_model_autumn/model_{}'.format(args.datapath,j)
        m.restore(save_path)
	print >> sys.stderr, "Restoring model {}/{}".format(j+1,num_ensembles)
        models.append(m)
    
    for ij, pmid in enumerate(pmids):
        print >> sys.stderr, "Processing {}/{} {}".format(ij+1,len(pmids),pmid)
        flattened_len = sum([ len(x) for x in tokens[pmid] ])
        proba_cumulative = np.zeros((flattened_len,len(labels)))

        for m in models:
            proba_cumulative += m.predict_proba(tokens[pmid],batch_size=20)
        
        y_pred = np.argmax(proba_cumulative,axis=1)
        
        label_list = [ labels[tag_idx] for tag_idx in y_pred ]
        
        prediction = []
        extra_data = []
        for idx,x in enumerate(tokens[pmid]):
            prediction.append(label_list[:len(x)])
            extra_data.append(tokens_data[pmid][idx])
            label_list = label_list[len(x):]
        
        print '###' + pmid
        print ''
        for idx_i,(line, tag) in enumerate(zip(tokens[pmid], prediction)):
            for idx_j,pair in enumerate(zip(line, tag)):
                tokenword = pair[0]
                predLabel = pair[1]
                trueLabel = extra_data[idx_i][idx_j]
                fullLine = [tokenword,trueLabel,predLabel]
                print ' '.join(fullLine)
                # print ' '.join(pair) + ' ' + extra_data[idx_i][idx_j]  # token truth_label pred_label
            
            print ''

def load_dataset(fname, shuffle=False):
    dataset = []
    with open(fname,'r') as f:
        dataset = [ x.split('\n') for x in f.read().split('\n\n') if x ]
    
    vocab = []
    output = []
    for x in dataset:
        
        tokens, labels = zip(*[ z.split(' ') for z in x if z ])
        for t in tokens:
            t = t.lower()
            if t not in vocab:
                vocab.append(t)
        
        output.append((tokens, labels))
    
    return output, vocab

if __name__ == '__main__':
    main(parser.parse_args())
