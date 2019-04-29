#!/usr/bin/python

import sys
import time
from collections import defaultdict, OrderedDict
import numpy
from utils import readConfig, readWordvectorsNumpy
from evaluation import evaluateModel
import random
import theano
import theano.tensor as T
import pickle
from layers import LeNetConvPoolLayer, HiddenLayer, LogisticRegression, CRF
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledExampleScheme
from iterationSchemes import ShuffledExampleSchemeBatch

class TrainModel:
  def __init__(self):
    self.relSet = []

  def getRelID(self,relName):
    return self.relSet.index(relName)

  def sgd_updates(self,params, cost, learning_rate, sqrt_norm_lim = 3):
      updates = []
      for param in params:
        gp = T.grad(cost, param)
        step = -1.0 * learning_rate * gp
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
          col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
          desired_norms = T.clip(col_norms, 0, sqrt_norm_lim)
          scale = desired_norms / (1e-7 + col_norms)
          updates.append((param, stepped_param * scale))
        else:
          updates.append((param, stepped_param))
      return updates

  def train(self, datafile, database_name, configfile, test_id2sent, test_id2arg2rel):
    config = readConfig(configfile)

    relationfile = config["basedir"] + config["relations_" + database_name]
    file = open(relationfile, 'r')
    for line in file:
      rel_arr = line.split(" ")
      self.relSet.append(rel_arr[1].replace('\n', '').replace('\r', ''))

    iterationSeed = -1
    if "iterationSeed" in config:
      iterationSeed = int(config["iterationSeed"])
      print("using " + str(iterationSeed) + " as seed for iteration scheme")
    pretrainedEmbeddings = False

    if "wordvectors" in config:
      wordvectorfile = config["basedir"] + config["wordvectors"]
      wordvectors, representationsize, words = readWordvectorsNumpy(wordvectorfile, isWord2vec = True)
      vocabsize = wordvectors.shape[0]
      pretrainedEmbeddings = True
    else:
      print("you have to specify a wordvector file in the config")
      exit()

    networkfile = config["basedir"] + config["net_"+database_name]
    output_model_file = networkfile
    lrateOrig = float(config["lrate"])
    print("using sgd with learning rate ", lrateOrig)
    batch_size = int(config["batchsize"])
    contextsize = int(config["contextsize"])
    entitysize = int(config["entitysize"])
    myLambda1 = 0
    if "lambda1" in config:
      myLambda1 = float(config["lambda1"])
    myLambda2 = 0
    if "lambda2" in config:
      myLambda2 = float(config["lambda2"])
    addInputSize = 1 # extra feature for hidden layer after convolution: name before filler?
    loss = "entropy"
    doCRF = False

    if "crf" in config:
      loss = "crf"
      doCRF = True
    print("using loss function: ", loss)
    examplesPerUpdate = None
    if "examplesPerUpdate" in config:
      examplesPerUpdate = int(config["examplesPerUpdate"])
    numPerBag = int(config["numPerBag"])
    numClasses = int(config["numClasses"])
    numClassesET = int(config["numClassesET"])

    nkernsContext = int(config["nkernsContext"])
    nkernsEntities = int(config["nkernsEntities"])
    hiddenUnits = int(config["hidden"])
    hiddenUnitsET = int(config["hiddenET"])
    filtersizeContext = int(config["filtersizeContext"])
    filtersizeEntities = int(config["filtersizeEntities"])
    kmaxContext = int(config["kmaxContext"])
    kmaxEntities = int(config["kmaxEntities"])

    time1 = time.time()

    ######## FUEL #################
    # Define "data_stream"
    # The names here (e.g. 'name1') need to match the names of the variables which
    #  are the roots of the computational graph for the cost.

    train_set = H5PYDataset(datafile, which_sets = ('train',), load_in_memory=True)
    dev_set = H5PYDataset(datafile, which_sets = ('dev',), load_in_memory=True)
    test_set = H5PYDataset(datafile, which_sets = ('test',), load_in_memory=True)
    numSamplesDev = dev_set.num_examples
    numSamplesTrain = train_set.num_examples
    numSamplesTest = test_set.num_examples

    print("got " + str(numSamplesTrain) + " training examples")
    numTrainingBatches = numSamplesTrain / batch_size
    print("got " + str(numSamplesDev) + " dev examples")
    print("got " + str(numSamplesTest) + " test examples")

    if iterationSeed != -1:
      data_stream = DataStream(train_set, iteration_scheme = ShuffledExampleSchemeBatch(numSamplesTrain, batch_size, iterationSeed))
    else:
      data_stream = DataStream(train_set, iteration_scheme = ShuffledExampleSchemeBatch(train_set.num_examples, batch_size))
    data_stream_dev = DataStream(dev_set, iteration_scheme=SequentialScheme(
                                           dev_set.num_examples, 1))
    data_stream_test = DataStream(test_set, iteration_scheme=SequentialScheme(
                                           test_set.num_examples, 1))
    numSamplesDev = dev_set.num_examples
    numSamplesTest = test_set.num_examples
    numSamplesTrain = (train_set.num_examples / batch_size) * batch_size
    ################################

    time2 = time.time()
    print("time for reading data: " + str(time2 - time1))

    # train network
    curSeed = 23455
    if "seed" in config:
      curSeed = int(config["seed"])
    rng = numpy.random.RandomState(curSeed)
    seed = rng.get_state()[1][0]
    print("seed: " + str(seed))

    x1 = T.imatrix('x1') # shape: (batchsize, numPerBag * contextsize) # left of e1
    x2 = T.imatrix('x2') # shape: (batchsize, numPerBag * contextsize) # right of e1
    x3 = T.imatrix('x3') # shape: (batchsize, numPerBag * contextsize) # left of e2
    x4 = T.imatrix('x4') # shape: (batchsize, numPerBag * contextsize) # right of e3
    y = T.imatrix('y') # shape: (batchsize, 1)
    y1ET = T.imatrix('y1ET') # shape: (batchsize, 1)
    y2ET = T.imatrix('y2ET') # shape: (batchsize, 1)
    e1 = T.imatrix('e1') # shape: (batchsize, entitysize)
    e2 = T.imatrix('e2') # shape: (batchsize, entitysize)
    numSamples = T.imatrix('numSamples') # shape: (batchsize, 1)
    lr = T.scalar('lr') # learning rate

    embeddings = theano.shared(numpy.array(wordvectors, dtype = theano.config.floatX)).dimshuffle(1,0)

    batchsizeVar = numSamples.shape[0]
    y_resh = y.reshape((batchsizeVar,)) # rel:e1->e2
    y1ET_resh = y1ET.reshape((batchsizeVar,))
    y2ET_resh = y2ET.reshape((batchsizeVar,))

    numSamples_resh = numSamples.reshape((batchsizeVar,))

    layers = []

    cnnContext = LeNetConvPoolLayer(rng = rng, filter_shape = (nkernsContext, 1, representationsize, filtersizeContext), poolsize = (1, kmaxContext))
    layers.append(cnnContext)
    if "middleContext" in config:
      hidden_in = nkernsContext * kmaxContext
    else:
      cnnEntities = LeNetConvPoolLayer(rng = rng, filter_shape = (nkernsEntities, 1, representationsize, filtersizeEntities), poolsize = (1, kmaxEntities))
      layers.append(cnnEntities)
      hidden_in = 2 * (2 * nkernsContext * kmaxContext + nkernsEntities * kmaxEntities)
    hiddenLayer = HiddenLayer(rng = rng, n_in = hidden_in, n_out = hiddenUnits)
    layers.append(hiddenLayer)
    hiddenLayerET = HiddenLayer(rng = rng, n_in = 2 * nkernsContext * kmaxContext + nkernsEntities * kmaxEntities, n_out = hiddenUnitsET)
    layers.append(hiddenLayerET)
    randomInit = False
    if doCRF:
      randomInit = True
    outputLayer = LogisticRegression(n_in = hiddenUnits, n_out = numClasses, rng = rng, randomInit = randomInit)
    layers.append(outputLayer)
    outputLayerET = LogisticRegression(n_in = hiddenUnitsET, n_out = numClassesET, rng = rng, randomInit = randomInit)
    layers.append(outputLayerET)
    if doCRF:
      crfLayer = CRF(numClasses = numClasses + numClassesET, rng = rng, batchsizeVar = batchsizeVar, sequenceLength = 3)
      layers.append(crfLayer)

    x1_resh = x1.reshape((batchsizeVar * numPerBag, contextsize))
    x1_emb = embeddings[:,x1_resh].dimshuffle(1, 0, 2)
    x1_emb = x1_emb.reshape((x1_emb.shape[0], 1, x1_emb.shape[1], x1_emb.shape[2]))
    x2_resh = x2.reshape((batchsizeVar * numPerBag, contextsize))
    x2_emb = embeddings[:,x2_resh].dimshuffle(1, 0, 2)
    x2_emb = x2_emb.reshape((x2_emb.shape[0], 1, x2_emb.shape[1], x2_emb.shape[2]))
    x3_resh = x3.reshape((batchsizeVar * numPerBag, contextsize))
    x3_emb = embeddings[:,x3_resh].dimshuffle(1, 0, 2)
    x3_emb = x3_emb.reshape((x3_emb.shape[0], 1, x3_emb.shape[1], x3_emb.shape[2]))
    x4_resh = x4.reshape((batchsizeVar * numPerBag, contextsize))
    x4_emb = embeddings[:,x4_resh].dimshuffle(1, 0, 2)
    x4_emb = x4_emb.reshape((x4_emb.shape[0], 1, x4_emb.shape[1], x4_emb.shape[2]))

    e1_resh = e1.reshape((batchsizeVar, entitysize))
    e1_emb = embeddings[:,e1_resh].dimshuffle(1, 0, 2)
    e1_emb = e1_emb.reshape((e1_emb.shape[0], 1, e1_emb.shape[1], e1_emb.shape[2]))
    e2_resh = e2.reshape((batchsizeVar, entitysize))
    e2_emb = embeddings[:,e2_resh].dimshuffle(1, 0, 2)
    e2_emb = e2_emb.reshape((e2_emb.shape[0], 1, e2_emb.shape[1], e2_emb.shape[2]))

    x1_rep = cnnContext.getOutput(x1_emb)
    x2_rep = cnnContext.getOutput(x2_emb)
    x3_rep = cnnContext.getOutput(x3_emb)
    x4_rep = cnnContext.getOutput(x4_emb)
    e1_rep = cnnEntities.getOutput(e1_emb)
    e2_rep = cnnEntities.getOutput(e2_emb)

    e1_rep_repeated = e1_rep.flatten(2).repeat(numPerBag, axis = 0)
    e2_rep_repeated = e2_rep.flatten(2).repeat(numPerBag, axis = 0)

    aroundE1 = T.concatenate([x1_rep.flatten(2), e1_rep_repeated, x2_rep.flatten(2)], axis = 1)
    aroundE2 = T.concatenate([x3_rep.flatten(2), e2_rep_repeated, x4_rep.flatten(2)], axis = 1)

    # entity typing:
    hiddenForE1 = hiddenLayerET.getOutput(aroundE1)
    hiddenForE2 = hiddenLayerET.getOutput(aroundE2)

    # relation classification:
    if "middleContext" in config:
      e1_emb_repeated = e1_emb.repeat(numPerBag, axis = 0)
      e2_emb_repeated = e2_emb.repeat(numPerBag, axis = 0)

      betweenE1E2 = cnnContext.getOutput(T.concatenate([e1_emb_repeated, x2_emb, e2_emb_repeated], axis = 3))

      betweenE1E2flatten = betweenE1E2.flatten(2)

      # to predict r1: between e1 and e2
      hiddenForR1 = hiddenLayer.getOutput(betweenE1E2flatten)

    else:
      # to predict r1: aroundE1 (x1 + e1 + x2) and aroundE2 (x2 + e2 + x3)
      hiddenForR1 = hiddenLayer.getOutput(T.concatenate([aroundE1,aroundE2], axis = 1))

    if doCRF:
      # scores for different classes for r1, r2 and r3
      scoresForR1 = outputLayer.getScores(hiddenForR1, numSamples, batchsizeVar)
      scoresForE1 = outputLayerET.getScores(hiddenForE1, numSamples, batchsizeVar)
      scoresForE2 = outputLayerET.getScores(hiddenForE2, numSamples, batchsizeVar)

      scores = T.zeros((batchsizeVar, 3, numClasses + numClassesET))
      scores = T.set_subtensor(scores[:,0,numClasses:], scoresForE1)
      scores = T.set_subtensor(scores[:,1,:numClasses], scoresForR1)
      scores = T.set_subtensor(scores[:,2,numClasses:], scoresForE2)
      y_conc = T.concatenate([y1ET + numClasses, y, y2ET + numClasses], axis = 1)
      cost = crfLayer.getCostAddLogWeights(scores, y_conc)
    else:
      cost = outputLayer.getCostMI(hiddenForR1, y_resh, numSamples, batchsizeVar) + outputLayerET.getCostMI(hiddenForE1, y1ET_resh, numSamples, batchsizeVar) + outputLayerET.getCostMI(hiddenForE2, y2ET_resh, numSamples, batchsizeVar)

    params = []
    for l in layers:
      params += l.params

    reg2 = 0.0
    reg1 = 0.0
    for p in params:
      if ".W" in p.name or "_W" in p.name:
        print("found W", p)
        reg2 += T.sum(p ** 2)
        reg1 += T.sum(abs(p))
    cost += myLambda2 * reg2
    cost += myLambda1 * reg1

    updates = self.sgd_updates(params, cost, learning_rate = lr)

    if doCRF:
      predictions_global = crfLayer.getPrediction(scores)
    else:
      predictions_y1 = outputLayer.getOutput(hiddenForR1, numSamples, batchsizeVar)
      predictions_et1 = outputLayerET.getOutput(hiddenForE1, numSamples, batchsizeVar)
      predictions_et2 = outputLayerET.getOutput(hiddenForE2, numSamples, batchsizeVar)

    train = theano.function([x1, x2, x3, x4, e1, e2, y, y1ET, y2ET, numSamples, lr], cost, updates = updates, on_unused_input='warn')
    if doCRF:
      getPredictions = theano.function([x1, x2, x3, x4, e1, e2, numSamples], predictions_global, on_unused_input='warn') # cut of padded begin and end
      getPredictionsR1 = None
      getPredictionsET1 = None
      getPredictionsET2 = None
    else:
      getPredictions = None
      getPredictionsR1 = theano.function([x1, x2, x3, x4, e1, e2, numSamples], predictions_y1,  on_unused_input='warn')
      getPredictionsET1 = theano.function([x1, x2, e1, numSamples], predictions_et1,  on_unused_input='warn')
      getPredictionsET2 = theano.function([x3, x4, e2, numSamples], predictions_et2,  on_unused_input='warn')

    ########## start training ###########################
    n_epochs = 15
    if "n_epochs" in config:
      n_epochs = int(config["n_epochs"])

    bestF1 = 0
    best_params = []
    best_epoch = 0
    epoch = 0
    lrate = lrateOrig
    while epoch < n_epochs:
      time1 = time.time()
      # train

      time1Train = time.time()
      for d in data_stream.get_epoch_iterator(as_dict = True):
        x1_numpy = d['x1']
        x2_numpy = d['x2']
        x3_numpy = d['x3']
        x4_numpy = d['x4']
        e1_numpy = d['e1']
        e2_numpy = d['e2']
        y_numpy = d['y']
        y1ET_numpy = d['y1ET']
        y2ET_numpy = d['y2ET']
        numSamples_numpy = numpy.ones_like(y1ET_numpy)

        cost_ij = train(x1_numpy, x2_numpy, x3_numpy, x4_numpy, e1_numpy, e2_numpy, y_numpy, y1ET_numpy, y2ET_numpy, numSamples_numpy, lrate)
        if numpy.isnan(cost_ij):
          print("ERROR: NAN in cost")
          epoch = n_epochs
          break

      time2Train = time.time()
      print("time for training: " + str(time2Train - time1Train))
      if epoch < n_epochs: # don't evaluate if cost was NAN
        # validate with table filling
        time1Eval = time.time()
        curF1, file = evaluateModel(data_stream_dev, epoch, doCRF, numClasses, numClassesET, getPredictions, getPredictionsR1, getPredictionsET1, getPredictionsET2)
        time2Eval = time.time()
        print("Average F1 over RE and ET: " + str(curF1))
        print("time for validation: " + str(time2Eval - time1Eval))
        if curF1 > bestF1:
          bestF1 = curF1
          best_epoch = epoch
          best_params = []
          for p in params:
            best_params.append(p.get_value(borrow=False))
        else:
          lrate = lrate * 0.5
          print("reducing learning rate to ", lrate)
          if lrate < 0.00001: # early stopping
            epoch = n_epochs
            break
        epoch += 1

      time2 = time.time()
      print("time for epoch: " + str(time2 - time1))
      print("")

    print("FINAL: result on dev: " + str(bestF1))
    # re-storing best model and saving it
    save_file = open(networkfile, 'wb')
    for p, bp in zip(params, best_params):
      p.set_value(bp, borrow=False)
      cPickle.dump(bp, save_file, -1)
    save_file.close()

    # validate best model on test
    f1_test, pred_relation_result = evaluateModel(data_stream_test, best_epoch, doCRF, numClasses, numClassesET, getPredictions, getPredictionsR1, getPredictionsET1, getPredictionsET2)
    print("FINAL: result on test: " + str(f1_test))

    pred_relation_file = config['basedir']+config['predicated_relation_' + database_name]
    output_file = pred_relation_file
    output_list = []

    for id in test_id2sent:
      sentence = test_id2sent[id]
      actual_relation = ''
      for pair in test_id2arg2rel[id]:
        actual_relation = test_id2arg2rel[id][pair]
      predicated_relation = ''
      if(id in pred_relation_result):
        predicated_relation = self.relSet[pred_relation_result[id]]
      output_list.append(sentence+", "+actual_relation+", "+predicated_relation)

    with open(pred_relation_file, 'w') as f:
      for item in output_list:
        f.write("%s\n" % item)

    return output_model_file, f1_test, output_file
