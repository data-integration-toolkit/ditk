#!/usr/bin/python

import sys
from random import shuffle
import random

#Read in the data
tweets = []
tokens = []

for line in open('ner.txt'):
    if line.strip() == '':
        tweets.append(tokens)
        tokens = []
    else:
        (word, label) = line.strip().split('\t')
        tokens.append((word,label))
random.seed(0)
shuffle(tweets)

trainOut = open('train', 'w')
devOut   = open('dev', 'w')
testOut   = open('test', 'w')

for i in range(len(tweets)):
    out = None
    if i < int(0.8 * len(tweets)):
        out = trainOut
    elif i < int(0.9 * len(tweets)):
        out = devOut
    else:
        out = testOut
    t = tweets[i]
    for w in t:
        out.write("%s\t%s\n" % w)
    out.write("\n")
