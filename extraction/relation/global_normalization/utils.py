#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import re
import random
import numpy
import codecs, sys

import io
import gzip

def readConfigBasic(configfile):
  config = {}
  # read config file
  f = open(configfile, 'r')
  for line in f:
    if "#" == line[0]:
      continue # skip commentars
    line = line.strip()
    parts = line.split('=')
    name = parts[0]
    value = parts[1]
    config[name] = value
  f.close()
  return config

def readConfig(configfile):
  config = readConfigBasic(configfile)
  return config

def readIndices(wordvectorfile, isWord2vec = True):
  indices = {}
  curIndex = 0
  indices["<empty>"] = curIndex
  curIndex += 1
  indices["<unk>"] = curIndex
  curIndex += 1
  if ".gz" in wordvectorfile:
    f = gzip.open(wordvectorfile, 'r')
  else:
    f = open(wordvectorfile,'r')
  count = 0
  for line in f:
    if isWord2vec:
      if count == 0:
        print("omitting first embedding line because of word2vec")
        count += 1
        continue
    parts = line.split()
    word = parts[0]
    indices[word] = curIndex
    curIndex += 1
  f.close()
  return indices

def readWordvectorsNumpy(wordvectorfile, isWord2vec = True):
  wordvectors = []
  words = []
  vectorsize = 0
  if ".gz" in wordvectorfile:
    f = gzip.open(wordvectorfile, 'r')
  else:
    f = open(wordvectorfile, 'r')
  count = 0
  for line in f:
    if isWord2vec:
      if count == 0:
        print("omitting first embedding line because of word2vec")
        count += 1
        continue
    parts = line.split()
    word = parts.pop(0) # ignore word string
    wordvectors.append([float(p) for p in parts])
    words.append(word)
    vectorsize = len(parts)
  f.close()
  # first entry: <empty> (zero) vector
  # second entry: <unk> (random) vector
  zeroVec = [0 for i in range(vectorsize)]
  random.seed(123456)
  randomVec = [random.uniform(-numpy.sqrt(1./len(wordvectors)), numpy.sqrt(1./len(wordvectors))) for i in range(vectorsize)]
  wordvectors.insert(0,randomVec)
  words.insert(0, "<unk>")
  wordvectors.insert(0, zeroVec)
  words.insert(0, "<empty>")

  wordvectorsNumpy = numpy.array(wordvectors)
  return wordvectorsNumpy, vectorsize, words

def getCoNNL_label2int(configfile,datasetName):
  config = readConfig(configfile)
  datafile = config["basedir"] + config["relations_"+datasetName]
  relSet=[]
  file = open(datafile, 'r')
  for line in file:
    rel_arr = line.split(" ")
    relSet.append(rel_arr[1].replace('\n',''))

  label2int = {}
  nerSet = ['L-Org', 'U-Loc', 'U-Peop', 'U-Org', 'B-Org', 'B-Other', 'I-Org', 'B-Peop', 'I-Loc', 'I-Peop', 'I-Other', 'L-Loc', 'U-Other', 'L-Other', 'B-Loc', 'L-Peop']
  index = 1 # index 0 = no ner / rel
  label2int['O'] = 0
  for n in nerSet:
    label2int[n] = index
    index += 1
  index = 1 # with two different softmax it's possible / even necessary to use the same integers again
  for r in relSet:
    label2int[r] = index
    index += 1
  return label2int

def getMatrixForContext(context, contextsize, wordindices):
  matrix = numpy.zeros(shape = (contextsize))
  i = 0
  nextIndex = 0
  while i < len(context):
    word = context[i]
    nextIndex = 0
    # current word
    if word != "<empty>":
      if not word in wordindices:
        if re.search(r'^\d+$', word):
          word = "0"
        if word.islower():
          word = word.title()
        else:
          word = word.lower()
      if not word in wordindices:
        word = "<unk>"
      curIndex = wordindices[word]
      matrix[i] = curIndex
    i += 1

  return matrix

def adaptNumSamplesTrain(numSamplesTrain, idTrain):
  while idTrain[numSamplesTrain] == idTrain[numSamplesTrain + 1]:
    numSamplesTrain += 1
  return numSamplesTrain + 1 # because we want the number of samples, not the index

def getRelID(relName,configfile,datasetName):
  config = readConfig(configfile)
  datafile = config["relations_" + datasetName]
  relSet = []
  file = open(datafile, 'r')
  for line in file:
    rel_arr = line.split(" ")
    relSet.append(rel_arr[1])
  return relSet.index(relName)

def getNerID(nerName):
  nerSet = ['O', 'Org', 'Loc', 'Peop', 'Other']
  return nerSet.index(nerName)

def cleanContext(context):
  c = " ".join(context)
  c = re.sub(r'\-LRB\-', '(', c)
  c = re.sub(r'\-RRB\-', ')', c)
  c = re.sub(r' COMMA ', ' , ', c)
  c = re.sub(r'(\S)(\W)$', '\\1 \\2', c)
  return c.split()

def reverse(x_in, x_len, numSamples, contentDim):
  x_rev = numpy.zeros(shape = (numSamples, contentDim))
  for i in range(numSamples):
    if x_len[i,0] > 0:
      # reverse context:
      x_rev[i,:x_len[i,0]] = x_in[i,x_len[i,0]-1::-1]
  return x_rev

def processPredictions(predictionsR1, probsR1):
    predictionsBatch = []
    for b in range(predictionsR1.shape[0]):
      predR1_b = predictionsR1[b]
      probR1_b = probsR1[b]
      maxPositiveProb = 0
      bestPrediction = 0
      for curPred, curProb in zip(predR1_b, probR1_b):
        if curPred > 0 and curProb > maxPositiveProb:
          maxPositiveProb = curProb
          bestPrediction = curPred
      predictionsBatch.append(bestPrediction)
    return predictionsBatch

def getReversedRel(rel):
  rev = numpy.zeros_like(rel)
  for b in range(rel.shape[0]):
    curRel = rel[b,0]
    if curRel == 0:
      rev[b,0] = 0
    elif curRel % 2 == 0:
      rev[b,0] = curRel - 1
    else:
      rev[b,0] = curRel + 1
  return rev

def getF1(allHypos, allRefs, numClasses, name = ""):
  class2precision = {}
  class2recall = {}
  class2f1 = {}
  class2tp = {}
  class2numHypo = {}
  class2numRef = {}
  for cl in range(numClasses): # initialize
    class2numHypo[cl] = 0
    class2numRef[cl] = 0
    class2tp[cl] = 0
    class2precision[cl] = 0
    class2recall[cl] = 0
    class2f1[cl] = 0
  for h, r in zip(allHypos, allRefs):
    if h >= numClasses:
      print("ERROR: prediction of " + str(h) + " but only " + str(numClasses) + " classes for " + name)
      h = 0
    class2numHypo[h] += 1
    class2numRef[r] += 1
    if h == r:
      class2tp[h] += 1
  sumF1 = 0
  for cl in range(1, len(class2numHypo.keys())):
    prec = 1.0
    numH = class2numHypo[cl]
    numR = class2numRef[cl]
    if numH > 0:
      prec = class2tp[cl] * 1.0 / numH
    class2precision[cl] = prec
    rec = 0.0
    if numR > 0:
      rec = class2tp[cl] * 1.0 / numR
    class2recall[cl] = rec
    f1 = 0.0
    if prec + rec > 0:
      f1 = prec * rec * 2.0 / (prec + rec)
    class2f1[cl] = f1
    sumF1 += f1
    print("Class " + str(cl) + ": numRef: " + str(numR) + ", numHypo: " + str(numH) + ", P = " + str(prec) + ", R = " + str(rec) + ", F1 = " + str(f1))
  macroF1 = sumF1 * 1.0 / (numClasses - 1)
  if name == "":
    print("Macro F1: " + str(macroF1))
  else:
    print("Macro F1 " + str(name) + ": " + str(macroF1))
  return macroF1

def getMajorityPrediction(types):
  hypos = [t[0] for t in types]
  refs = [t[1] for t in types]
  assert len(set(refs)) == 1
  sortedHypos = sorted([(hypos.count(e), e) for e in set(hypos)], key=lambda x:x[0], reverse=True)
  elems = [h[1] for h in sortedHypos]
  counts = [h[0] for h in sortedHypos]
  if len(counts) == 1 or counts[0] != counts[1]: # easy case
    return elems[0], refs[0]
  # select most common class among hypos with highest votes
  bestCounts = 0
  i = 1
  while i < len(counts) and counts[i] == counts[0]:
    bestCounts = i
    i += 1
  bestElems = elems[:bestCounts + 1]
  # order of ET classes according to frequency:
  # 1. loc: 2
  # 2. per: 3
  # 3. org: 1
  # 4. other: 4
  for mostFreq in [2, 3, 1, 4]:
    if mostFreq in bestElems:
      return mostFreq, refs[0]
  return 0, refs[0]

def getRelaxedPredictionEntityType(predictions, refs):
  assert len(set(refs)) == 1
  ref = refs[0]
  if ref in predictions: # prediction is considered as correct
    return ref, ref
  else:
    return predictions[0], ref # just pick random prediction

def getPredictionRelation(predictions, refs, relationEvaluationMethod):
  assert len(set([r[2] for r in refs])) == 1
  ref = refs[0][2]
  if relationEvaluationMethod == "relaxed": # hypo is correct if one of the hypos is correct
    hypos = [h[2] for h in predictions]
    if ref in hypos:
      return ref, ref
    else:
      return hypos[0], ref # random prediction
  else: # hypo is prediction in cell with last token of entities
    maximumE1 = max([h[0] for h in predictions])
    maximumE2 = max([h[1] for h in predictions])
    for h in predictions:
      if h[0] == maximumE1 and h[1] == maximumE2:
        return h[2], ref
    # default return, should never happen
    return predictions[0], ref

def mergeREPredictionsWithOldIndices(curSentence_entityPair2relations, newIndex2oldIndex):
  curSentence_pair2predictions = {}
  curSentence_pair2refs = {}
  for ent1, ent2 in curSentence_entityPair2relations:
    oldIndex1a, oldIndex1b = newIndex2oldIndex[ent1].split("_")
    oldIndex2a, oldIndex2b = newIndex2oldIndex[ent2].split("_")
    if oldIndex1a == oldIndex2a:
      continue # this is entity typing, not relation classification
    if not (oldIndex1a, oldIndex2a) in curSentence_pair2predictions:
      curSentence_pair2predictions[(oldIndex1a, oldIndex2a)] = []
      curSentence_pair2refs[(oldIndex1a, oldIndex2a)] = []
    for rIndex in range(len(curSentence_entityPair2relations[(ent1, ent2)])):
      curSentence_pair2predictions[(oldIndex1a, oldIndex2a)].append((oldIndex1b, oldIndex2b, curSentence_entityPair2relations[(ent1, ent2)][rIndex][0]))
      curSentence_pair2refs[(oldIndex1a, oldIndex2a)].append((oldIndex1b, oldIndex2b, curSentence_entityPair2relations[(ent1, ent2)][rIndex][1]))
  return curSentence_pair2predictions, curSentence_pair2refs

def mergeETPredictionsWithOldIndices(curSentence_entity2types, newIndex2oldIndex):
  curSentence_ent2majorityPredictions = {}
  curSentence_ent2refs = {}
  for ent in curSentence_entity2types:
    majorityPrediction = getMajorityPrediction(curSentence_entity2types[ent])
    oldIndex = int(newIndex2oldIndex[ent].split("_")[0])
    if not oldIndex in curSentence_ent2majorityPredictions:
      curSentence_ent2majorityPredictions[oldIndex] = []
      curSentence_ent2refs[oldIndex] = []
    curSentence_ent2majorityPredictions[oldIndex].append(majorityPrediction[0])
    curSentence_ent2refs[oldIndex].append(majorityPrediction[1])
  return curSentence_ent2majorityPredictions, curSentence_ent2refs

