#!/usr/bin/python

from fuel.schemes import BatchScheme
from picklable_itertools import iter_, imap
from picklable_itertools.extras import partition_all
import numpy

class ShuffledExampleSchemeBatch(BatchScheme):
  def __init__(self, examples, batch_size, seed = 987654):
    super(ShuffledExampleSchemeBatch, self).__init__(examples, batch_size)
    self.batch_size = batch_size
    numpy.random.seed(seed)

  def get_request_iterator(self):
    indices = list(self.indices)
    # shuffle indices
    indicesShuffled = []
    permutation = numpy.random.permutation(len(indices))
    return imap(list, partition_all(self.batch_size, permutation))

