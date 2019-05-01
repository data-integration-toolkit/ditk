import nrekit
import numpy as np
import tensorflow as tf
import sys
import os
import json
import sklearn.metrics
import matplotlib
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CNN():

    def __init__(self):
        pass

    def read_dataset(self, input_file, *args, **kwargs):  
        """
        Reads a dataset to be used for training
         
         Note: The child file of each member overrides this function to read dataset 
         according to their data format.
         
        Args:
            input_file: Filepath with list of files to be read
        Returns: 
            (optional):Data from file
        """
        f = open("data/nyt/trainfile.txt", "r")
        thisdict =  {
        #'sentence': 'Bill Gates is the founder of Microsoft .',
                #'head': {'word': 'Bill Gates', 'id': 'm.03_3d',
                #'tail': {'word': 'Microsoft', 'id': 'm.07dfk', 
                #'relation': 'founder'
        }
        head ={}
        tail ={}
        outerlist=[]

        for x in f:
            temp = x.split('\t')
            #print(x.split('\t'))
            thisdict["sentence"] = temp[0]
            head["word"] = temp[1]
            head["id"]= temp[1].replace(" ","")
            tail["word"] = temp[5]
            tail["id"]= temp[5].replace(" ","")
            thisdict["head"] = head
            thisdict["tail"] = tail
            thisdict["relation"] = temp[9].rstrip()
            
            outerlist.append(thisdict)
            
            
            #print(temp[0], temp[1])
            #print(x)
        with open('data/nyt/train.json', 'w') as outfile:
            json.dump(outerlist, outfile)

        print "Processed trainging file"

        f = open("data/nyt/testfile.txt", "r")
        thisdict =  {
        #'sentence': 'Bill Gates is the founder of Microsoft .',
                #'head': {'word': 'Bill Gates', 'id': 'm.03_3d',
                #'tail': {'word': 'Microsoft', 'id': 'm.07dfk', 
                #'relation': 'founder'
        }
        head ={}
        tail ={}
        outerlist=[]

        for x in f:
            temp = x.split('\t')
            #print(x.split('\t'))
            thisdict["sentence"] = temp[0]
            head["word"] = temp[1]
            head["id"]= temp[1].replace(" ","")
            tail["word"] = temp[5]
            tail["id"]= temp[5].replace(" ","")
            thisdict["head"] = head
            thisdict["tail"] = tail
            thisdict["relation"] = temp[9].rstrip()
            
            outerlist.append(thisdict)
            
            
            #print(temp[0], temp[1])
            #print(x)
        with open('data/nyt/test.json', 'w') as outfile:
            json.dump(outerlist, outfile)

        print "Processed testing file"

    def data_preprocess(self,input_data, *args, **kwargs):
        """
         (Optional): For members who do not need preprocessing. example: .pkl files 
         A common function for a set of data cleaning techniques such as lemmatization, count vectorizer and so forth.
        Args: 
            input_data: Raw data to tokenize
        Returns:
            Formatted data for further use.
        """
        pass 


    def tokenize(self, input_data ,ngram_size=None, *args, **kwargs):  
        """
        Tokenizes dataset using Stanford Core NLP(Server/API)
        Args:
            input_data: str or [str] : data to tokenize
            ngram_size: mention the size of the token combinations, default to None
        Returns:
            tokenized version of data
        """
        pass


    def train(self, train_data, *args, **kwargs):  
        """
        Trains a model on the given training data
        
         Note: The child file of each member overrides this function to train data 
         according to their algorithm.
         
        Args:
            train_data: post-processed data to be trained.
        
        Returns: 
            (Optional) : trained model in applicable formats.
             None: if the model is stored internally.
        """

        class model(nrekit.framework.re_model):
            encoder = "cnn"
            selector = "ave"

            def __init__(self, train_data_loader, batch_size, max_length=120):
                nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
                self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
                
                # Embedding
                with tf.name_scope('embedding'):
                    x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

                # Encoder
                with tf.name_scope('encoder'):
                    if model.encoder == "pcnn":
                        x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
                        x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
                    elif model.encoder == "cnn":
                        x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
                        x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
                    elif model.encoder == "rnn":
                        x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
                        x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
                    elif model.encoder == "birnn":
                        x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
                        x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
                    else:
                        raise NotImplementedError

                # Selector
                with tf.name_scope('selector'):
                    if model.selector == "att":
                        self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
                        self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
                    elif model.selector == "ave":
                        self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot, keep_prob=0.5)
                        self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot, keep_prob=1.0)
                        self._test_logit = tf.nn.softmax(self._test_logit)
                    elif model.selector == "one":
                        self._train_logit, train_repre = nrekit.network.selector.bag_one(x_train, self.scope, self.label, self.rel_tot, True, keep_prob=0.5)
                        self._test_logit, test_repre = nrekit.network.selector.bag_one(x_test, self.scope, self.label, self.rel_tot, False, keep_prob=1.0)
                        self._test_logit = tf.nn.softmax(self._test_logit)
                    elif model.selector == "cross_max":
                        self._train_logit, train_repre = nrekit.network.selector.bag_cross_max(x_train, self.scope, self.rel_tot, keep_prob=0.5)
                        self._test_logit, test_repre = nrekit.network.selector.bag_cross_max(x_test, self.scope, self.rel_tot, keep_prob=1.0)
                        self._test_logit = tf.nn.softmax(self._test_logit)
                    else:
                        raise NotImplementedError
                
                # Classifier
                with tf.name_scope('classifier'):
                    self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot, weights_table=self.get_weights())
         
            def loss(self):
                return self._loss

            def train_logit(self):
                return self._train_logit

            def test_logit(self):
                return self._test_logit

            def get_weights(self):
                with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
                    print("Calculating weights_table...")
                    _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
                    for i in range(len(self.train_data_loader.data_rel)):
                        _weights_table[self.train_data_loader.data_rel[i]] += 1.0 
                    _weights_table = 1 / (_weights_table ** 0.05)
                    weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
                    print("Finish calculating")
                return weights_table

        dataset_name = 'nyt'
        dataset_dir = os.path.join('./data', dataset_name)
        if not os.path.isdir(dataset_dir):
            raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

        # The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
        train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'), 
                                                                os.path.join(dataset_dir, 'word_vec.json'),
                                                                os.path.join(dataset_dir, 'rel2id.json'), 
                                                                mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                                shuffle=True)
        test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'), 
                                                               os.path.join(dataset_dir, 'word_vec.json'),
                                                               os.path.join(dataset_dir, 'rel2id.json'), 
                                                               mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                               shuffle=False)

        framework = nrekit.framework.re_framework(train_loader, test_loader)

        model.encoder = "cnn"
        model.selector = "ave"

        framework.train(model, model_name=dataset_name + "_" + model.encoder + "_" + model.selector, max_epoch=1, ckpt_dir="checkpoint", gpu_nums=1)


    def predict(self, test_data, entity_1 = None, entity_2= None,  trained_model = None, *args, **kwargs):   
        """
        Predict on the trained model using test data
        Args:
              entity_1, entity_2: for some models, given an entity, give the relation most suitable 
            test_data: test the model and predict the result.
            trained_model: the trained model from the method - def train().
                          None if store trained model internally.
        Returns:
              probablities: which relation is more probable given entity1, entity2 
                  or 
            relation: [tuple], list of tuples. (Eg - Entity 1, Relation, Entity 2) or in other format 
        """
        class model(nrekit.framework.re_model):
            encoder = "cnn"
            selector = "ave"

            def __init__(self, train_data_loader, batch_size, max_length=120):
                nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
                self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
                
                # Embedding
                x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

                # Encoder
                if model.encoder == "pcnn":
                    x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
                    x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
                elif model.encoder == "cnn":
                    x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
                    x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
                elif model.encoder == "rnn":
                    x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
                    x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
                elif model.encoder == "birnn":
                    x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
                    x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
                else:
                    raise NotImplementedError

                # Selector
                if model.selector == "att":
                    self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
                    self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
                elif model.selector == "ave":
                    self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot, keep_prob=0.5)
                    self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot, keep_prob=1.0)
                    self._test_logit = tf.nn.softmax(self._test_logit)
                elif model.selector == "max":
                    self._train_logit, train_repre = nrekit.network.selector.bag_maximum(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
                    self._test_logit, test_repre = nrekit.network.selector.bag_maximum(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
                    self._test_logit = tf.nn.softmax(self._test_logit)
                else:
                    raise NotImplementedError
                
                # Classifier
                self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot, weights_table=self.get_weights())
         
            def loss(self):
                return self._loss

            def train_logit(self):
                return self._train_logit

            def test_logit(self):
                return self._test_logit

            def get_weights(self):
                with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
                    print("Calculating weights_table...")
                    _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
                    for i in range(len(self.train_data_loader.data_rel)):
                        _weights_table[self.train_data_loader.data_rel[i]] += 1.0 
                    _weights_table = 1 / (_weights_table ** 0.05)
                    weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
                    print("Finish calculating")
                return weights_table

        dataset_name = 'nyt'
        dataset_dir = os.path.join('./data', dataset_name)
        if not os.path.isdir(dataset_dir):
            raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

        # The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
        train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'), 
                                                                os.path.join(dataset_dir, 'word_vec.json'),
                                                                os.path.join(dataset_dir, 'rel2id.json'), 
                                                                mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                                shuffle=True)
        test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'), 
                                                               os.path.join(dataset_dir, 'word_vec.json'),
                                                               os.path.join(dataset_dir, 'rel2id.json'), 
                                                               mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                               shuffle=False)

        framework = nrekit.framework.re_framework(train_loader, test_loader)

        model.encoder = "cnn"
        model.selector = "ave"

        auc, pred_result = framework.test(model, ckpt="./checkpoint/" + dataset_name + "_" + model.encoder + "_" + model.selector, return_result=True)

        with open('./test_result/' + dataset_name + "_" + model.encoder + "_" + model.selector + "_pred.json", 'w') as outfile:
            json.dump(pred_result, outfile)

        ##### Output required format #########
        with open("data/nyt/rel2id.json") as f:
            rel2id = json.load(f)

        id2rel = {}
        for key, value in rel2id.iteritems():
            id2rel[value-1] = key

        N_realtion = len(id2rel)
        #with open("test_result/nyt_pcnn_ave_pred.json") as f:
        #    test_result = json.load(f)
        test_result = pred_result
        prev = ""
        pair_orders = []
        for entity_pair in test_result:
            if entity_pair["entpair"] != prev:
                pair_orders.append(entity_pair["entpair"])
                prev = entity_pair["entpair"]

        relation_probs = {}
        for entity_pair in test_result:
            key = entity_pair["entpair"]
            probs = relation_probs.get(key,np.zeros((N_realtion,)))
            if np.isnan(entity_pair["score"]):
                prob = 0
            else:
                prob = entity_pair["score"]
            probs[entity_pair["relation"]-1] = prob
            relation_probs[key] = probs

        pred_ralations = []
        for key in pair_orders:
            pred_ralations.append(id2rel[np.argmax(relation_probs[key])])

        with open("data/nyt/test.json") as f:
            test_input = json.load(f)
        print pred_ralations
        assert (len(test_input)) == len(pred_ralations) 

        output = []
        for single_input, pred_ralation in zip(test_input, pred_ralations):
            to_append = (single_input["sentence"].encode('utf-8'), single_input["head"]["word"].encode('utf-8'), 
                         single_input["tail"]["word"].encode('utf-8'), pred_ralation, single_input["relation"].encode('utf-8'))
            output.append(to_append)

        with open("test_result/common_output.tsv", "w") as f:
            for element in output:
                normal_str = [str(word) for word in element]
                f.write("\t".join(normal_str) + "\n")

        return output

    def evaluate(self, input_data, trained_model = None, *args, **kwargs):
        """
        Evaluates the result based on the benchmark dataset and the evauation metrics  [Precision,Recall,F1, or others...]
         Args:
             input_data: benchmark dataset/evaluation data
             trained_model: trained model or None if stored internally 
        Returns:
            performance metrics: tuple with (p,r,f1) or similar...
        """
        models = "nyt_pcnn_ave"
        for model in models:
            x = np.load(os.path.join(result_dir, model +'_x' + '.npy')) 
            y = np.load(os.path.join(result_dir, model + '_y' + '.npy'))
            f1 = (2 * x * y / (x + y + 1e-20)).max()
            auc = sklearn.metrics.auc(x=x, y=y)
            #plt.plot(x, y, lw=2, label=model + '-auc='+str(auc))
            plt.plot(x, y, lw=2, label=model)
            print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1))
            print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300], (y[100] + y[200] + y[300]) / 3))
           
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.3, 1.0])
        plt.xlim([0.0, 0.4])
        plt.title('Precision-Recall')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'pr_curve'))


if __name__ == '__main__':
    #to change dataset from NYT to something else like DDI or semeval. find nyt in this file and change to dataset name 
    # instatiate the class
    myModel = CNN()
    print "Reading dataset"
    myModel.read_dataset("foo")
    data = None
    print "Training"
    myModel.train(None)
    print "Predicting"
    #predictions = myModel.predict(None)  # generate predictions! output format will be same for everyone
    #print predictions
    #print "Evaluating"
    #myModel.evaluate(None)