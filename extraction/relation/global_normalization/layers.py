#!/usr/bin/python

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import theano.sandbox.neighbours as TSN

class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh, name=""):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        self.activation = activation

        if name != "":
          prefix = name
        else:
          prefix = "mlp_"

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'b', borrow=True)

        self.W = W
        self.b = b

        # parameters of the model
        self.params = [self.W, self.b]

    def getOutput(self, input):
        lin_output = T.dot(input, self.W) + self.b
        output = (lin_output if self.activation is None
                       else self.activation(lin_output))
        return output


#########################################################################################

class LogisticRegression(object):

    def __init__(self, n_in, n_out, W = None, b = None, rng = None, randomInit = False):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        self.numClasses = n_out

        if W == None:
          if randomInit:
            name = 'softmax_random_W'
            fan_in = n_in
            fan_out = n_out
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(value=numpy.asarray(
                                rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
                                dtype=theano.config.floatX),
                                name=name, borrow=True)
          else:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='softmax_W', borrow=True)
        else:
          self.W = W

        self.params = [self.W]

        if b == None:
          # initialize the baises b as a vector of n_out 0s
          self.b = theano.shared(value=numpy.zeros((n_out,),
                                    dtype=theano.config.floatX),
                                    name='softmax_b', borrow=True)
        else:
          self.b = b
        self.params.append(self.b)

    def getMask(self, batchsize, maxSamplesInBag, samplesInBags):
      # mask entries outside of bags
      mask = T.zeros((batchsize, maxSamplesInBag))
      maskAcc, _ = theano.scan(fn = lambda b, m: T.set_subtensor(m[b,:samplesInBags[b,0]], 1),
                          outputs_info=mask, sequences=T.arange(batchsize))
      mask = maskAcc[-1]
      mask2 = mask.repeat(self.numClasses, axis = 1).reshape((batchsize, maxSamplesInBag, self.numClasses))
      return mask2

    def nll_mi(self, x, y, samplesInBags, batchsize):
      self.p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
      maxSamplesInBag = self.p_y_given_x.shape[0] / batchsize
      self.p_y_given_x = self.p_y_given_x.reshape((batchsize, maxSamplesInBag, self.p_y_given_x.shape[1]))
      mask = self.getMask(batchsize, maxSamplesInBag, samplesInBags)

      self.p_y_given_x_masked = self.p_y_given_x * T.cast(mask, theano.config.floatX)
      maxpredvec = T.max(self.p_y_given_x_masked, axis = 1)
      batch_cost_log = T.log(maxpredvec)[T.arange(y.shape[0]), y]

      numberOfValidExamples = T.sum(T.cast(mask[:,:,0], theano.config.floatX))
      return -T.sum(batch_cost_log) / numberOfValidExamples

    def getCostMI(self, x, y, samplesInBags, batchsize, rankingParam=2, m_minus = 0.5, m_plus = 2.5):
      return self.nll_mi(x, y, samplesInBags, batchsize)

    def getScores(self, x, samplesInBags, batchsize):
      return self.getScores_softmax(x, samplesInBags, batchsize)

    def getOutput(self, x, samplesInBags, batchsize):
      return self.getOutput_softmax(x, samplesInBags, batchsize)

    def getScores_softmax(self, x, samplesInBags, batchsize):
      predictions = T.dot(x, self.W) + self.b
      maxSamplesInBag = predictions.shape[0] / batchsize
      predictions = predictions.reshape((batchsize, maxSamplesInBag, predictions.shape[1]))
      mask = self.getMask(batchsize, maxSamplesInBag, samplesInBags)
      predictions_masked = predictions * T.cast(mask, theano.config.floatX)
      maxpredvec = T.max(predictions_masked, axis = 1)
      return maxpredvec

    def getOutput_softmax(self, x, samplesInBags, batchsize):
      self.p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
      maxSamplesInBag = self.p_y_given_x.shape[0] / batchsize
      self.p_y_given_x = self.p_y_given_x.reshape((batchsize, maxSamplesInBag, self.p_y_given_x.shape[1]))
      mask = self.getMask(batchsize, maxSamplesInBag, samplesInBags)
      self.p_y_given_x_masked = self.p_y_given_x * T.cast(mask, theano.config.floatX)
      argmaxpredvec = T.argmax(self.p_y_given_x_masked, axis = 2)
      maxpredvec = T.max(self.p_y_given_x_masked, axis = 2)
      return [argmaxpredvec, maxpredvec]


####################################################################################

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def preparePooling(self, conv_out):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling
      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      neighborsArgSorted = neighborsArgSorted
      return neighborsForPooling, neighborsArgSorted

    def kmaxPooling(self, conv_out, k):
      neighborsForPooling, neighborsArgSorted = self.preparePooling(conv_out)
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape,
                image_shape=None)

    def __init__(self, rng, filter_shape, image_shape = None, W = None, b = None, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.filter_shape = filter_shape
        self.poolsize = poolsize

        if W == None:
          fan_in = numpy.prod(self.filter_shape[1:])
          fan_out = (self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) /
              numpy.prod(self.poolsize))

          W_bound = numpy.sqrt(6. / (fan_in + fan_out))
          # the convolution weight matrix
          self.W = theano.shared(numpy.asarray(
             rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
             dtype=theano.config.floatX), name='conv_W',
                               borrow=True)
        else:
          self.W = W

        if b == None:
          # the bias is a 1D tensor -- one bias per output feature map
          b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
          self.b = theano.shared(value=b_values, name='conv_b', borrow=True)
        else:
          self.b = b

        # store parameters of this layer
        self.params = [self.W, self.b]

    def getOutput(self, input):

        # convolve input feature maps with filters
        conv_out = self.convStep(input, self.W)

        #self.conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        k = self.poolsize[1]
        self.pooledkmax = self.kmaxPooling(conv_out, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

###################################################################################

class CRF:
    # Code from https://github.com/glample/tagger/blob/master/model.py
    # but extended to support mini-batches

    def log_sum_exp(self, x, axis=None):
      """
      Sum probabilities in the log-space.
      """
      xmax = x.max(axis=axis, keepdims=True)
      xmax_ = x.max(axis=axis)
      return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def recurrence(self, obs, previous):
        previous = previous.dimshuffle(0, 1, 'x')
        obs = obs.dimshuffle(0, 'x', 1)
        return self.log_sum_exp(previous + obs + self.transitions.dimshuffle('x', 0, 1), axis=1)

    def recurrence_viterbi(self, obs, previous):
        previous = previous.dimshuffle(0, 1, 'x')
        obs = obs.dimshuffle(0, 'x', 1)
        scores = previous + obs + self.transitions.dimshuffle('x', 0, 1)
        out = scores.max(axis=1)
        return out

    def recurrence_viterbi_returnBest(self, obs, previous):
        previous = previous.dimshuffle(0, 1, 'x')
        obs = obs.dimshuffle(0, 'x', 1)
        scores = previous + obs + self.transitions.dimshuffle('x', 0, 1)
        out = scores.max(axis=1)
        out2 = scores.argmax(axis=1)
        return out, out2

    def forward(self, observations, viterbi=False, return_alpha=False, return_best_sequence=False):
      """
      Takes as input:
        - observations, sequence of shape (batch_size, n_steps, n_classes)
      Probabilities must be given in the log space.
      Compute alpha, matrix of size (batch_size, n_steps, n_classes), such that
      alpha[:, i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
      Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
      """
      assert not return_best_sequence or (viterbi and not return_alpha)

      def recurrence_bestSequence(b):
        sequence_b, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][b][-1]), 'int32'),
            sequences=T.cast(alpha[1][b,::-1], 'int32')
        )
        return sequence_b

      initial = observations[:,0]

      if viterbi:
        if return_best_sequence:
          alpha, _ = theano.scan(
            fn=self.recurrence_viterbi_returnBest,
            outputs_info=(initial, None),
            sequences=[observations[:,1:].dimshuffle(1,0,2)] # shuffle to get a sequence over time, not over batches
          )
          alpha[0] = alpha[0].dimshuffle(1,0,2) # shuffle back
          alpha[1] = alpha[1].dimshuffle(1,0,2)
        else:
          alpha, _ = theano.scan(
            fn=self.recurrence_viterbi,
            outputs_info=initial,
            sequences=[observations[:,1:].dimshuffle(1,0,2)] # shuffle to get a sequence over time, not over batches
          )
          alpha = alpha.dimshuffle(1,0,2) # shuffle back
      else:
        alpha, _ = theano.scan(
          fn=self.recurrence,
          outputs_info=initial,
          sequences=[observations[:,1:].dimshuffle(1,0,2)] # shuffle to get a sequence over time, not over batches
        )
        alpha = alpha.dimshuffle(1,0,2) # shuffle back

      if return_alpha:
        return alpha
      elif return_best_sequence:
        batchsizeVar = alpha[0].shape[0]
        sequence, _ = theano.scan(
           fn=recurrence_bestSequence,
           outputs_info = None,
           sequences=T.arange(batchsizeVar)
        )
        sequence = T.concatenate([sequence[:,::-1], T.argmax(alpha[0][:,-1], axis = 1).reshape((batchsizeVar, 1))], axis = 1)
        return sequence, alpha[0]
      else:
        if viterbi:
          return alpha[:,-1,:].max(axis=1)
        else:
          return self.log_sum_exp(alpha[:,-1,:], axis=1)

    def __init__(self, numClasses, rng, batchsizeVar, sequenceLength = 3):
      self.numClasses = numClasses

      shape_transitions = (numClasses + 2, numClasses + 2) # +2 because of start id and end id
      drange = numpy.sqrt(6.0 / numpy.sum(shape_transitions))
      self.transitions = theano.shared(value = numpy.asarray(rng.uniform(low = -drange, high = drange, size = shape_transitions), dtype = theano.config.floatX), name = 'transitions')

      self.small = -1000 # log for very small probability
      b_s = numpy.array([[self.small] * numClasses + [0, self.small]]).astype(theano.config.floatX)
      e_s = numpy.array([[self.small] * numClasses + [self.small, 0]]).astype(theano.config.floatX)
      self.b_s_theano = theano.shared(value = b_s).dimshuffle('x', 0, 1)
      self.e_s_theano = theano.shared(value = e_s).dimshuffle('x', 0, 1)

      self.b_s_theano = self.b_s_theano.repeat(batchsizeVar, axis = 0)
      self.e_s_theano = self.e_s_theano.repeat(batchsizeVar, axis = 0)

      self.s_len = sequenceLength

      self.debug1 = self.e_s_theano

      self.params = [self.transitions]

    def getObservations(self, scores):
      batchsizeVar = scores.shape[0]
      observations = T.concatenate([scores, self.small * T.cast(T.ones((batchsizeVar, self.s_len, 2)), theano.config.floatX)], axis = 2)
      observations = T.concatenate([self.b_s_theano, observations, self.e_s_theano], axis = 1)
      return observations

    def getPrediction(self, scores):
      observations = self.getObservations(scores)
      prediction = self.forward(observations, viterbi=True, return_best_sequence=True)
      return prediction

    def getCost(self, scores, y_conc):
      batchsizeVar = scores.shape[0]
      observations = self.getObservations(scores)

      # score from classes
      scores_flattened = scores.reshape((scores.shape[0] * scores.shape[1], scores.shape[2]))
      y_flattened = y_conc.flatten(1)

      real_path_score = scores_flattened[T.arange(batchsizeVar * self.s_len), y_flattened]
      real_path_score = real_path_score.reshape((batchsizeVar, self.s_len)).sum(axis = 1) 

      # score from transitions
      b_id = theano.shared(value=numpy.array([self.numClasses], dtype=numpy.int32)) # id for begin
      e_id = theano.shared(value=numpy.array([self.numClasses + 1], dtype=numpy.int32)) # id for end
      b_id = b_id.dimshuffle('x', 0).repeat(batchsizeVar, axis = 0)
      e_id = e_id.dimshuffle('x', 0).repeat(batchsizeVar, axis = 0)
      
      padded_tags_ids = T.concatenate([b_id, y_conc, e_id], axis=1)

      real_path_score2, _ = theano.scan(fn = lambda m: self.transitions[padded_tags_ids[m,T.arange(self.s_len+1)], padded_tags_ids[m,T.arange(self.s_len + 1) + 1]].sum(), sequences = T.arange(batchsizeVar), outputs_info = None)

      real_path_score += real_path_score2
      all_paths_scores = self.forward(observations)
      self.debug1 = real_path_score
      cost = - T.mean(real_path_score - all_paths_scores)
      return cost

    def getCostAddLogWeights(self, scores, y_conc):
      batchsizeVar = scores.shape[0]
      observations = self.getObservations(scores)

      # score from classes
      scores_flattened = scores.reshape((scores.shape[0] * scores.shape[1], scores.shape[2]))
      y_flattened = y_conc.flatten(1)

      real_path_score = scores_flattened[T.arange(batchsizeVar * self.s_len), y_flattened]
      real_path_score = real_path_score.reshape((batchsizeVar, self.s_len)).sum(axis = 1)

      # score from transitions
      b_id = theano.shared(value=numpy.array([self.numClasses], dtype=numpy.int32)) # id for begin
      e_id = theano.shared(value=numpy.array([self.numClasses + 1], dtype=numpy.int32)) # id for end
      b_id = b_id.dimshuffle('x', 0).repeat(batchsizeVar, axis = 0)
      e_id = e_id.dimshuffle('x', 0).repeat(batchsizeVar, axis = 0)

      padded_tags_ids = T.concatenate([b_id, y_conc, e_id], axis=1)

      real_path_score2, _ = theano.scan(fn = lambda m: self.transitions[padded_tags_ids[m,T.arange(self.s_len+1)], padded_tags_ids[m,T.arange(self.s_len + 1) + 1]].sum(), sequences = T.arange(batchsizeVar), outputs_info = None)

      real_path_score += real_path_score2
      all_paths_scores = self.forward(observations)
      self.debug1 = real_path_score
      cost = - T.mean(real_path_score - all_paths_scores) 
      return cost

