#!/usr/bin/env python
import sys, os, random, pickle, json, codecs, time
import numpy as np
from model import BiLSTM
import sklearn.metrics as skm
import argparse
#import evaluation

# parser = argparse.ArgumentParser(description='Train BiLSTM on a given dataset')
# parser.add_argument('--datapath', dest='datapath', type=str,
#                     default='../corpus_train', 
#                     help='path to the datasets')
# parser.add_argument('--embeddings', dest='embeddings_path', type=str,
#                     default='../embeddings/PubMed-w2v.txt', 
#                     help='path to the testing dataset')
# parser.add_argument('--optimizer', dest='optimizer', type=str,
#                     default='default', 
#                     help='choose the optimizer: default, rmsprop, adagrad, adam.')
# parser.add_argument('--batch-size', dest='batch_size', type=int, 
#                     default=20, help='number of instances in a minibatch')
# parser.add_argument('--num-iterations', dest='num_iterations', type=int, 
#                     default=5000, help='number of iterations')
# parser.add_argument('--num-it-per-ckpt', dest='num_it_per_ckpt', type=int, 
#                     default=100, help='number of iterations per checkpoint')
# parser.add_argument('--learning-rate', dest='learning_rate', type=str,
#                     default='default', help='learning rate')
# parser.add_argument('--embedding-factor', dest='embedding_factor', type=float,
#                     default=1.0, help='learning rate multiplier for embeddings')
# parser.add_argument('--decay', dest='decay_rate', type=float,
#                     default=0.95, help='exponential decay for learning rate')
# parser.add_argument('--keep-prob', dest='keep_prob', type=float,
#                     default=0.7, help='dropout keep rate')
# parser.add_argument('--num-cores', dest='num_cores', type=int, 
#                     default=4, help='seed for training')
# parser.add_argument('--seed', dest='seed', type=int, 
#                     default=2, help='seed for training')

num_ensembles = 10

# def main(args):
def model_train(datapath,embeddings_path,optimizer,batch_size,num_iterations,num_it_per_ckpt,learning_rate,embedding_factor,decay_rate,keep_prob,num_cores,seed):

    # datapath = 'corpus_train/'  # data directory
    # embeddings_path = 'embeddings/PubMed-w2v.txt'  # path to embeddings file. expects 200d embeddings. REQUIRED
    # optimizer = 'adam'  # optimizer to use, in 'default, rmsprop, adagrad, adam'. set to adam for now
    # batch_size = 16  # batch size for training
    # num_iterations = 5000  # number of iterations to run [an iteration is a parameter/weight update cycle from optimizer on singel batch] PER MODEL [i.e. single component of ensemble]
    # num_it_per_ckpt = 100  # after this many iterations, save a checkpoint model file and consider it for best as this model [single ensemble]
    # learning_rate = 'default'  # do not change
    # embedding_factor = 1.0  # do not change
    # decay_rate = 0.95  # do not change?
    # keep_prob = 0.7  # do not change?
    # num_cores = 4  # number of cores to exploit parallelism
    # seed = 2  # random seed to use for pseudorandom initializations and reproducibility

    print "Running BiLSTM model"
    print args
    random.seed(seed)
    
    trainset = []
    devset = []

    print >> sys.stderr, "Loading dataset.."
    assert(os.path.isdir(datapath))
    
    word_vocab = []
    for fname in sorted(os.listdir(datapath)):
        if os.path.isdir(fname): 
            continue
        
        #if fname.endswith('train.ner.txt'):
        if fname.endswith('.ppi.txt'):
            print fname
            dataset, vocab = load_dataset(os.path.join(datapath,fname))
            word_vocab += vocab
            trainset += dataset
        
            print >> sys.stderr, "Loaded {} instances with a vocab size of {} from {}".format(len(dataset),len(vocab),fname)
    
    print "Loaded {} instances from data set".format(len(trainset))
    
    word_vocab = sorted(set(word_vocab))
    vocab_cache = os.path.join(datapath,'word_vocab.ner.txt')
    with open(vocab_cache,'w') as f:
        print "Saved vocab to", vocab_cache
        pickle.dump(word_vocab,f)
    
    if not os.path.exists('{}/scratch'.format(datapath)):
        os.mkdir('{}/scratch'.format(datapath))

    embeddings = load_embeddings(embeddings_path, word_vocab, 200)
    
    labels = ['B-MISC','I-MISC','O']
    
    model_name = 'saved_model_autumn'
    # if not os.path.exists('{}/scratch'.format(datapath)):
    #     os.mkdir('{}/scratch'.format(datapath))
            
    if os.path.exists('{}/{}'.format(datapath,model_name)):
        os.rename('{}/{}'.format(datapath,model_name),
            '{}/{}_{}'.format(datapath,model_name,int(time.time())))
        
    os.mkdir('{}/{}'.format(datapath,model_name))
    
    for j in range(num_ensembles):
        m = BiLSTM(labels=labels,
                    word_vocab=word_vocab,
                    word_embeddings=embeddings,
                        optimizer=optimizer,
                        embedding_size=200, 
                        char_embedding_size=32,
                        lstm_dim=200,
                        num_cores=num_cores,
                        embedding_factor=embedding_factor,
                        learning_rate=learning_rate,
                        decay_rate=decay_rate,
                        dropout_keep=keep_prob)
        
        training_samples = random.sample(trainset,len(trainset)/2)
        
        cut = int(0.8 * len(training_samples))
        X_train, y_train = zip(*training_samples[:cut]) 
        X_dev, y_dev = zip(*training_samples[cut:]) 
        
        print "Training on {}, tuning on {}".format(len(X_train),len(X_dev))
        
        m.fit(X_train, y_train, X_dev, y_dev,
                num_iterations=num_iterations,
                num_it_per_ckpt=num_it_per_ckpt,
                batch_size=batch_size,
                seed=j, fb2=True)
        
        save_path = '{}/{}/model_{}'.format(datapath,model_name,j)
        m.save(save_path)
        print "Saved model {} to {}".format(j,save_path)

def load_embeddings(fname, vocab, dim=200):
    if not os.path.exists('./scratch'):
        os.mkdir('./scratch')
    vocabset = set(vocab)
    cached = './scratch/embeddings_{}.npy'.format(abs(hash(' '.join(vocab))))
    
    if not os.path.exists(cached):
        weight_matrix = np.random.uniform(-0.05, 0.05, (len(vocab),dim)).astype(np.float32)
        ct = 0
        
        ctime = time.time()
        print 'Loading embeddings..',
        with codecs.open(fname, encoding='utf-8') as f:
#            data = f.read()
#        print '{}s'.format(int(time.time()-ctime))
        
            ctime = time.time()
            print 'Organizing embeddings..',
            lookup = {}
#        for line in data.split('\n'):
            for line in f:
                if line.strip() == '':
                    continue
            
                word, vec = line.split(u' ', 1)
                if word in vocabset:
                    # print 'adding word'
                    lookup[word] = vec
            print 'done f'
        print '{}s'.format(int(time.time()-ctime))
            
        for word in vocab:
            if word not in lookup:
                continue
            
            vec = lookup[word]
            idx = vocab.index(word)
            vec = np.array(vec.split(), dtype=np.float32)
            weight_matrix[idx,:dim] = vec[:dim]
            ct += 1
            if ct % 33 == 0:
                sys.stdout.write('Vectorizing embeddings {}/{}   \r'.format(ct, len(vocab)))
        
        print "Loaded {}/{} embedding vectors".format(ct, len(vocab))
        np.save(cached,weight_matrix)
    else:
        weight_matrix = np.load(cached)
    
    print "Loaded weight matrix {}..".format(weight_matrix.shape)
    
    return weight_matrix

def load_dataset(fname, shuffle=False):
    dataset = []
    with open(fname,'r') as f:
        dataset = [ x.split('\n') for x in f.read().split('\n\n') if x and not x.startswith('#') ]
    
    vocab = []
    output = []
    for x in dataset:
        tokens, labels = zip(*[ z.split(' ')[:2] for z in x if z ])

        for t in tokens:
            t = t.lower()
            if t not in vocab:
                vocab.append(t)
        
        output.append((tokens, labels))
    
    return output, vocab

# if __name__ == '__main__':
#     main(parser.parse_args())
