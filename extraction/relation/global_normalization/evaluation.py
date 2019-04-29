#!/usr/bin/python

import numpy
import json
from utils import getF1, processPredictions, getMajorityPrediction, getRelaxedPredictionEntityType, getPredictionRelation, mergeREPredictionsWithOldIndices, mergeETPredictionsWithOldIndices


def evaluateModel(datastream, epoch, doCRF, numClasses, numClassesET, getPredictions, getPredictionsR1, getPredictionsET1, getPredictionsET2):
  # validate on dev data
  allHyposRE = []
  allRefsRE = []
  allHyposET = []
  allRefsET = []
  curSentId = -1
  curSentence_entity2types = {}
  result=dict()
  for d in datastream.get_epoch_iterator(as_dict = True):
    x1_numpy = d['x1']
    x2_numpy = d['x2']
    x3_numpy = d['x3']
    x4_numpy = d['x4']
    e1_numpy = d['e1']
    e2_numpy = d['e2']
    y_numpy = d['y']
    y1ET_numpy = d['y1ET']
    y2ET_numpy = d['y2ET']
    sent_id_numpy = d['sent_id']
    e1_id_numpy = d['e1_id']
    e2_id_numpy = d['e2_id']
    numSamples_numpy = numpy.ones_like(y_numpy)
    if doCRF:
      predictions, probs = getPredictions(x1_numpy, x2_numpy, x3_numpy, x4_numpy, e1_numpy, e2_numpy, numSamples_numpy)
      predictions_rel = predictions[:,2::2] # cut off begin and end padding
      for b in range(predictions_rel.shape[0]):
        allHyposRE.append(predictions_rel[b][0])
        allRefsRE.append(y_numpy[b][0])
      predictions_et = predictions[:,1::2] - numClasses # cut off begin and end padding and account for vector concatenation with RE scores
      for b in range(predictions.shape[0]):
        if curSentId == -1:
          curSentId = sent_id_numpy[b][0]
        if sent_id_numpy[b][0] == curSentId:
          pass # only append below
        else:
          for ent in curSentence_entity2types:
            majorityPrediction = getMajorityPrediction(curSentence_entity2types[ent])
            allHyposET.append(majorityPrediction[0])
            allRefsET.append(majorityPrediction[1])
          curSentence_entity2types = {}
          curSentId = sent_id_numpy[b][0]
        key1 = e1_id_numpy[b][0]
        key2 = e2_id_numpy[b][0]
        if not key1 in curSentence_entity2types:
          curSentence_entity2types[key1] = []
        if not key2 in curSentence_entity2types:
          curSentence_entity2types[key2] = []
        curSentence_entity2types[key1].append((predictions_et[b][0],y1ET_numpy[b][0]))
        curSentence_entity2types[key2].append((predictions_et[b][1],y2ET_numpy[b][0]))
    else:
      predictionsR1, probsR1 = getPredictionsR1(x1_numpy, x2_numpy, x3_numpy, x4_numpy, e1_numpy, e2_numpy, numSamples_numpy)
      curBatchPredictionsR1 = processPredictions(predictionsR1, probsR1)
      allHyposRE.extend(curBatchPredictionsR1)
      allRefsRE.extend(y_numpy.flatten().tolist())
      predictionsET1, probsET1 = getPredictionsET1(x1_numpy, x2_numpy, e1_numpy, numSamples_numpy)
      curBatchPredictionsET1 = processPredictions(predictionsET1, probsET1)
      predictionsET2, probsET2 = getPredictionsET2(x3_numpy, x4_numpy, e2_numpy, numSamples_numpy)
      curBatchPredictionsET2 = processPredictions(predictionsET2, probsET2)
      for b in range(len(curBatchPredictionsET1)):
        if curSentId == -1:
          curSentId = sent_id_numpy[b][0]
        if sent_id_numpy[b][0] == curSentId:
          pass # only append below
        else:
          for ent in curSentence_entity2types:
            majorityPrediction = getMajorityPrediction(curSentence_entity2types[ent])
            allHyposET.append(majorityPrediction[0])
            allRefsET.append(majorityPrediction[1])
          curSentence_entity2types = {}
          curSentId = sent_id_numpy[b][0]
        key1 = e1_id_numpy[b][0]
        key2 = e2_id_numpy[b][0]
        if not key1 in curSentence_entity2types:
          curSentence_entity2types[key1] = []
        if not key2 in curSentence_entity2types:
          curSentence_entity2types[key2] = []
        curSentence_entity2types[key1].append((curBatchPredictionsET1[b],y1ET_numpy[b][0]))
        curSentence_entity2types[key2].append((curBatchPredictionsET2[b],y2ET_numpy[b][0]))

    if y_numpy[b][0]!=0:
      result[sent_id_numpy[b][0]]=predictions_rel[b][0]

  # also include predictions from last sentence
  for ent in curSentence_entity2types:
    majorityPrediction = getMajorityPrediction(curSentence_entity2types[ent])
    allHyposET.append(majorityPrediction[0])
    allRefsET.append(majorityPrediction[1])


  print("Validation after epoch " + str(epoch) + ":")
  f1_rel = getF1(allHyposRE, allRefsRE, numClasses, name="RE")
  f1_et = getF1(allHyposET, allRefsET, numClassesET, name = "ET")
  f1 = 0.5 * (f1_rel + f1_et)

  return f1,result


