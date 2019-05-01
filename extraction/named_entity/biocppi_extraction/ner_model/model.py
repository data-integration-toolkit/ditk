import sys, os, random, pickle, re, time, string
import numpy as np
import tensorflow as tf
import sklearn.metrics as skm
#import evaluation

class BiLSTM(object):
    def __init__(self, labels, word_vocab, 
                    word_embeddings=None,
                    embedding_size=300,
                    char_embedding_size=32,
                    lstm_dim=200, 
                    optimizer='default', learning_rate='default', 
                    embedding_factor = 1.0, decay_rate=1.0, 
                    dropout_keep=0.8, 
                    num_cores=4):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.inter_op_parallelism_threads=num_cores
        config.intra_op_parallelism_threads=num_cores 
        
        self.sess = tf.Session(config=config)
        self.labels = []
        self.embedding_size = embedding_size
        self.char_embedding_size = char_embedding_size
        self.char_feature_embedding_size = 8
        self.optimizer = optimizer
        self.decay = decay_rate
        
        if optimizer == 'default':
            self.optimizer = 'sgd'
        else:
            self.optimizer = optimizer
        
        if learning_rate is not 'default':
            self.lrate = float(learning_rate)
        else:
            if self.optimizer in ['adam','rmsprop']:
                self.lrate = 0.001
            elif self.optimizer == 'adagrad':
                self.lrate = 0.5
            elif self.optimizer == 'sgd':
                self.lrate = 0.2
            else:
                raise Exception('Unknown optimizer {}'.format(optimizer))
        
        self.embedding_factor = embedding_factor
        self.rnn_dim = lstm_dim
        self.dropout_keep = dropout_keep
        self.labels = labels
        self.word_vocab = word_vocab
        self.word_embeddings = word_embeddings
        self.char_buckets = 255
        self.window_size = 3
        self.num_filters = 50
        
        self._compile()
    
    def _compile(self):
        with self.sess.as_default(): 
            import tensorflow_fold as td
        
        output_size = len(self.labels)
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0),shape=None)

        fshape = (self.window_size * (self.char_embedding_size + self.char_feature_embedding_size), self.num_filters)
        filt_w3 = tf.Variable(tf.random_normal(fshape, stddev=0.05))

        def CNN_Window3(filters):
            return td.Function(lambda a, b, c: cnn_operation([a,b,c],filters))

        def cnn_operation(window_sequences,filters):
            windows = tf.concat(window_sequences,axis=-1)
            products = tf.multiply(tf.expand_dims(windows,axis=-1),filters)
            return tf.reduce_sum(products,axis=-2)

        char_emb = td.Embedding(num_buckets=self.char_buckets, 
                                num_units_out=self.char_embedding_size)
        
        cnn_layer = (td.NGrams(self.window_size) 
                        >> td.Map(CNN_Window3(filt_w3)) 
                        >> td.Max())

        # --------------------- Character Features -----------------------
        
        def charfeature_lookup(c):
            if c in string.lowercase:
                return 0
            elif c in string.uppercase:
                return 1
            elif c in string.punctuation:
                return 2
            else:
                return 3

        char_input = td.Map(td.InputTransform(lambda c: ord(c.lower())) 
                            >> td.Scalar('int32') >> char_emb)
                            
        char_features = td.Map(td.InputTransform(charfeature_lookup) 
                            >> td.Scalar(dtype='int32') 
                            >> td.Embedding(num_buckets=4,
                                            num_units_out=self.char_feature_embedding_size))

        charlevel = (td.InputTransform(lambda s: ['~'] + [ c for c in s ] + ['~']) 
                        >> td.AllOf(char_input,char_features) >> td.ZipWith(td.Concat()) 
                        >> cnn_layer)        

        # ---------------------- Word Features ---------------------------
        
        word_emb = td.Embedding(num_buckets=len(self.word_vocab),
                                num_units_out=self.embedding_size,
                                initializer=self.word_embeddings)
        
        wordlookup = lambda w: (self.word_vocab.index(w.lower()) 
                                if w.lower() in self.word_vocab else 0)
        
        wordinput = (td.InputTransform(wordlookup) 
                        >> td.Scalar(dtype='int32') 
                        >> word_emb)
        
        def wordfeature_lookup(w):
            if re.match('^[a-z]+$',w):
                return 0
            elif re.match('^[A-Z][a-z]+$',w):
                return 1
            elif re.match('^[A-Z]+$',w):
                return 2
            elif re.match('^[A-Za-z]+$',w):
                return 3
            else:
                return 4
        
        wordfeature = (td.InputTransform(wordfeature_lookup) 
                        >> td.Scalar(dtype='int32') 
                        >> td.Embedding(num_buckets=5,
                                num_units_out=32))
        
        #------------------------ Output Layer ---------------------------
        
        rnn_fwdcell = td.ScopedLayer(tf.contrib.rnn.LSTMCell(
                        num_units=self.rnn_dim), 'lstm_fwd')
        fwdlayer = td.RNN(rnn_fwdcell) >> td.GetItem(0)
        
        rnn_bwdcell = td.ScopedLayer(tf.contrib.rnn.LSTMCell(
                        num_units=self.rnn_dim), 'lstm_bwd')
        bwdlayer = (td.Slice(step=-1) >> td.RNN(rnn_bwdcell) 
                    >> td.GetItem(0) >> td.Slice(step=-1))
        
        rnn_layer = td.AllOf(fwdlayer, bwdlayer) >> td.ZipWith(td.Concat())
        
        output_layer = td.FC(output_size, 
                             input_keep_prob=self.keep_prob, 
                             activation=None)
        
        wordlevel = td.AllOf(wordinput,wordfeature) >> td.Concat()
        
        network = (td.Map(td.AllOf(wordlevel,charlevel) >> td.Concat()) 
                        >> rnn_layer 
                        >> td.Map(output_layer) 
                        >> td.Map(td.Metric('y_out'))) >> td.Void()
    
        groundlabels = td.Map(td.Vector(output_size,dtype=tf.int32) 
                                >> td.Metric('y_true')) >> td.Void()
    
        self.compiler = td.Compiler.create((network, groundlabels))
        
        self.y_out = self.compiler.metric_tensors['y_out']
        self.y_true = self.compiler.metric_tensors['y_true']
        
        self.y_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y_out,labels=self.y_true))

        self.y_prob = tf.nn.softmax(self.y_out)
        self.y_true_idx = tf.argmax(self.y_true,axis=-1)
        self.y_pred_idx = tf.argmax(self.y_prob,axis=-1)
        
        self.y_pred = tf.one_hot(self.y_pred_idx,depth=output_size,dtype=tf.int32)
        
        epoch_step = tf.Variable(0, trainable=False)
        self.epoch_step_op = tf.assign(epoch_step, epoch_step+1)
            
        lrate_decay = tf.train.exponential_decay(self.lrate, epoch_step, 1, self.decay)
            
        if self.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=lrate_decay)
        elif self.optimizer == 'adagrad':
            self.opt = tf.train.AdagradOptimizer(learning_rate=lrate_decay,
                                                initial_accumulator_value=1e-08)
        elif self.optimizer == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=lrate_decay,
                                                 epsilon=1e-08)
        elif self.optimizer == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=lrate_decay)
        else:
            raise Exception(('The optimizer {} is not in list of available ' 
                            + 'optimizers: default, adam, adagrad, rmsprop.')
                            .format(self.optimizer))
        
        # apply learning multiplier on on embedding learning rate
        embeds = [word_emb.weights]
        grads_and_vars = self.opt.compute_gradients(self.y_loss)
        found = 0
        for i, (grad, var) in enumerate(grads_and_vars):
            if var in embeds:
                found += 1
                grad = tf.scalar_mul(self.embedding_factor, grad)
                grads_and_vars[i] = (grad, var)
        
        assert found == len(embeds)  # internal consistency check
        self.train_step = self.opt.apply_gradients(grads_and_vars)        
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)
    
    def _onehot(self, y, categories):
        y_onehot = np.zeros((len(y),len(categories)))
        for i in range(len(y)):
            y_onehot[i,categories.index(y[i])] = 1
        
        return y_onehot
    
    def _train_minibatches(self,minibatches):
        mavg_loss = None
        
        for k, minibatch in enumerate(minibatches):
            varl = [self.train_step, self.y_loss, self.y_pred_idx, self.y_true_idx]
            minibatch[self.keep_prob] = self.dropout_keep
            _, ym_loss, ym_pred, ym_true = self.sess.run(varl, minibatch)
            
            if mavg_loss is None:
                mavg_loss = ym_loss
            else:
                mavg_loss = 0.8 * mavg_loss + 0.2 * ym_loss
            
            sys.stdout.write(" >> training {}/{} loss={:.7f}  \r".format(
                k+1,len(minibatches),mavg_loss))
            sys.stdout.flush()
    
    def fit(self, X, y, X_dev, y_dev, num_iterations = 10000, 
            num_it_per_ckpt = 100, batch_size = 20, seed = 1, fb2 = False):
        random.seed(seed)
        session_id = int(time.time())
        trainset = zip(X, [ self._onehot(l,self.labels) for l in y ])
        devset = zip(X_dev, [ self._onehot(l,self.labels) for l in y_dev ])
        print "Target labels: {}".format(self.labels)
        
        train_split = trainset
        valid_split = devset

        print "{}/{} in training/validation set".format(len(train_split),len(valid_split))
        print "Using batch_size of {}".format(batch_size)
        
        trainsp = random.sample(train_split,min(len(X)/2,200))
        trainfd = self.compiler.build_feed_dict(trainsp)
        #valfd = self.compiler.build_feed_dict(valid_split)
        
        best_epoch = 0
        best_model = None
        best_score = 0
        epochs_since_best = 0
        for i in range(num_iterations/num_it_per_ckpt):
            estart = time.time()
            
            minibatches = []
            for k in range(num_it_per_ckpt): # checkpoint every 100 iterations
                pool = random.sample(train_split,batch_size)
                minibatches.append(self.compiler.build_feed_dict(pool))
            
            self._train_minibatches(minibatches)
            self.sess.run(self.epoch_step_op)
            
            loss, yt_pred, yt_true = self.sess.run([self.y_loss, self.y_pred_idx, self.y_true_idx], trainfd)
            f, precision, recall = self.fscore(yt_pred,yt_true,fb2=fb2)
            
            #yv_pred, yv_true = self.sess.run([self.y_pred_idx, self.y_true_idx], valfd)
            #vf, vprecision, vrecall = self.fscore(yv_pred,yv_true)
            vf, vprecision, vrecall = self.evaluate(X_dev,y_dev,fb2=fb2)
            
            save_marker = ''
            if vf >= best_score:
                best_model = 'scratch/model-{}-{}-e{}-s{}.ckpt'.format(
                    session_id,type(self).__name__.lower(),i,seed)
                
                best_epoch, best_score = i, vf
                self.saver.save(self.sess, best_model)
                save_marker = '*'
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                
            elapsed = int(time.time() - estart)
            emin, esec = elapsed / 60, elapsed % 60
            print "ckpt {} bsize={} loss {} fit {:.2%} val {:.2%}/{:.2%}/{:.2%}[{}m{}s] {}".format(i, 
                batch_size, loss, f, vf, vprecision, vrecall, emin, esec, save_marker)
            
            if epochs_since_best > 10:
                print "Stopping early from lack of improvements.."
                break
        
        if best_model is None:
            print "WARNING: NO GOOD FIT"
        
        self.saver.restore(self.sess, best_model)
        print "Fitted to model from chkpt {} with score {} at {}".format(best_epoch,best_score,best_model)
    
    def save(self, model_path):
        self.saver.save(self.sess, model_path)
    
    def restore(self, model_path):
        tf.reset_default_graph()
        self.saver.restore(self.sess, model_path)
    
    def predict(self, X, batch_size = 10):
        dummy_labels = []
        for x in X:
            dummy_seq = [ np.zeros(len(self.labels)) ] * len(x)
            dummy_labels.append(dummy_seq)
            
        testset_all = zip(X,dummy_labels)
        
        prediction_idx = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self.compiler.build_feed_dict(testset)
            prediction_idx.append(self.sess.run(self.y_pred_idx, testfd))
        
        label_list = []
        for tag_idx in np.concatenate(prediction_idx):
            label_list.append(self.labels[tag_idx])
        
        prediction = []
        for x in X:
            prediction.append(label_list[:len(x)])
            label_list = label_list[len(x):]
        
        return prediction
    
    def predict_proba(self, X, batch_size = 100):
        dummy_labels = []
        for x in X:
            dummy_seq = [ np.zeros(len(self.labels)) ] * len(x)
            dummy_labels.append(dummy_seq)
            
        testset_all = zip(X,dummy_labels)
        
        y_prob_list = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self.compiler.build_feed_dict(testset)
            y_prob_list.append(self.sess.run(self.y_prob, testfd))
        
        return np.concatenate(y_prob_list,axis=0)

    def evaluate(self,X,y, batch_size = 100, macro = False, fb2 = False):
        testset_all = zip(X, [ self._onehot(l,self.labels) for l in y])
        
        y_pred_idx = []
        y_true_idx = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self.compiler.build_feed_dict(testset)
            yp, yt = self.sess.run([self.y_pred_idx,self.y_true_idx], testfd)
            y_pred_idx += list(yp)
            y_true_idx += list(yt)
        
        return self.fscore(y_pred_idx,y_true_idx,macro=macro,fb2=fb2)
        
    def fscore(self,y_pred,y_true, macro=False, fb2=False):
        avg_meth = 'micro'
        if macro:
            avg_meth = 'macro'
        
        if fb2:
            beta = 2
        else:
            beta = 1
        
        labels = [ i for i in range(len(self.labels)) if self.labels[i] not in ['false','False', False, 'O' ] ]
        f = skm.fbeta_score(y_true, y_pred, average=avg_meth,labels=labels, beta=beta)
        p = skm.precision_score(y_true, y_pred, average=avg_meth,labels=labels)
        r = skm.recall_score(y_true, y_pred, average=avg_meth,labels=labels)
        return f, p ,r

