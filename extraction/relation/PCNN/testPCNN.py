import nrekit
import numpy as np
import tensorflow as tf
import sys
import os
import json
import sklearn.metrics

if __name__ == '__main__':
    # instatiate the class
    myModel = PCNN()
    print "Reading dataset"
    dataset_name = "nyt"
    encoder="pcnn"
    selector="ave"
    myModel.read_dataset(dataset_name)
    print "Training"
    myModel.train(encoder, selector, dataset_name, epoch=1)
    print "Predicting"
    predictions = myModel.predict(encoder, selector, dataset_name)  # generate predictions! output format will be same for everyone
    print "Evaluating"
    myModel.evaluate(encoder, selector, dataset_name)