from ner import NER
import argparse

class CollaboNet(Ner):

    def __init__(self):
        pass


    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        Converts test data into common format for evaluation [i.e. same format as predict()]
        This added step/layer of abstraction is required due to the refactoring of read_dataset_traint()
        and read_dataset_test() back to the single method of read_dataset() along with the requirement on
        the format of the output of predict() and therefore the input format requirement of evaluate(). Since
        individuals will implement their own format of data from read_dataset(), this is the layer that
        will convert to proper format for evaluate().
        Args:
            data: data in proper [arbitrary] format for train or test. [i.e. format of output from read_dataset]
        Returns:
            ground_truth: [tuple,...], i.e. list of tuples. [SAME format as output of predict()]
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
        Raises:
            None
        """
        pass

    def run_util(self, file_dict, dataset_name, lr_decay, epoch):
        parser = argparse.ArgumentParser()
        parser.add_argument('--guidee_data', type=str, help='data name', default='name')
        parser.add_argument('--pretrained', type=int, help='pretrained STM expName', default=0)
        parser.add_argument('--ncbi', action='store_true', help='include ncbi data', default=False)
        parser.add_argument('--jnlpba', action='store_true', help='include jnlpba data', default=False)
        parser.add_argument('--bc2', action='store_true', help='include bc2gm data', default=False)
        parser.add_argument('--bc4', action='store_true', help='include bc4chemd data', default=False)
        parser.add_argument('--bc5_disease', action='store_true', help='include bc5-disease data', default=False)
        parser.add_argument('--bc5_chem', action='store_true', help='include bc5-chem data', default=False)
        parser.add_argument('--bc5', action='store_true', help='include bc5cdr data', default=False)
        parser.add_argument('--tensorboard', action='store_true', help='single flag [default]False', default=False)
        parser.add_argument('--epoch', type=int, help='max epoch', default=epoch)
        parser.add_argument('--num_class', type=int, help='result class bio(3) [default]biolu(5)', default=5)
        parser.add_argument('--ce_dim', type=int, help='char embedding dim', default=30)
        parser.add_argument('--clwe_dim', type=int, help='char level word embedding dim', default=200)
        parser.add_argument('--clwe_method', type=str, help='clwe method: CNN biLSTM', default='CNN')
        parser.add_argument('--batch_size', type=int, help='batch size', default=10)
        parser.add_argument('--hidden_size', type=int, help='lstm hidden layer size', default=300)
        parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
        parser.add_argument('--lr_decay', type=float, help='learning rate dacay rate', default=lr_decay)
        parser.add_argument('--lr_pump', action='store_true', help='do lr_pump', default=False)
        parser.add_argument('--loss_weight', type=float, help='loss weight between CRF, LSTM', default=1)
        parser.add_argument('--fc_method', type=str, help='fc method', default='normal')
        parser.add_argument('--mlp_layer', type=int, help='num highway layer ', default=1)
        parser.add_argument('--char_maxlen', type=int, help='char max length', default=49)
        parser.add_argument('--embdropout', type=float, help='input embedding dropout_rate', default=0.5)
        parser.add_argument('--lstmdropout', type=float, help='lstm output dropout_rate', default=0.3)
        parser.add_argument('--seed', type=int, help='seed value', default=0)
        args = parser.parse_args()

        self.args = args

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tf verbose off(info, warning)

        # seed initialize
        expName = setExpName()
        self.expName = expName
        if args.seed != 0:
            seedV = int(args.seed % 100000)
        else:
            try:
                tempSeed = int(expName)
            except:
                tempSeed = int(expName[:12])
            seedV = int(tempSeed % 100000)

        random.seed(seedV)
        np.random.seed(seedV)
        print tf.set_random_seed(seedV)

        # gpu setting
        gpu_config = tf.ConfigProto(device_count={'GPU': 1})  # only use GPU no.1
        gpu_config.gpu_options.allow_growth = True  # only use required resource(memory)
        gpu_config.gpu_options.per_process_gpu_memory_fraction = 1  # restrict to 100%
        self.gpu_config = gpu_config


        ID2wordVecIdx, wordVec2LineNo, wordEmbedding = input_wordVec()
        ID2char = pickle.load(open('data/ID2char.pickle', 'rb'))

        m_train = file_dict['train']
        m_dev = file_dict['dev']
        m_test = file_dict['test']

        self.m_train = m_train
        self.m_dev = m_dev
        self.m_test = m_test

        modelDict = OrderedDict()
        if args.ncbi:
            ncbi_args = deepcopy(args)
            ncbi_args.guidee_data = 'NCBI'
            modelDict['NCBI'] = {'args': ncbi_args}
        if args.jnlpba:
            jnl_args = deepcopy(args)
            jnl_args.guidee_data = 'JNLPBA'
            modelDict['JNLPBA'] = {'args': jnl_args}
        if args.bc2:
            bc2_args = deepcopy(args)
            bc2_args.guidee_data = 'BC2GM'
            modelDict['BC2GM'] = {'args': bc2_args}
        if args.bc4:
            bc4_args = deepcopy(args)
            bc4_args.guidee_data = 'BC4CHEMD'
            modelDict['BC4CHEMD'] = {'args': bc4_args}
        if args.bc5_chem:
            bc5c_args = deepcopy(args)
            bc5c_args.guidee_data = 'BC5CDR-chem'
            modelDict['BC5CDR-chem'] = {'args': bc5c_args}
        if args.bc5_disease:
            bc5d_args = deepcopy(args)
            bc5d_args.guidee_data = 'BC5CDR-disease'
            modelDict['BC5CDR-disease'] = {'args': bc5d_args}
        if args.bc5:
            bc5_args = deepcopy(args)
            bc5_args.guidee_data = 'BC5CDR'
            modelDict['BC5CDR'] = {'args': bc5_args}

        modelStart = time.time()
        modelClass = Model(args, wordEmbedding, seedV)

        for dataSet in modelDict:
            modelDict[dataSet]['summery'] = dict()
            modelDict[dataSet]['CLWE'] = modelClass.clwe(args=modelDict[dataSet]['args'], ID2char=ID2char)
            modelDict[dataSet]['WE'] = modelClass.we(args=modelDict[dataSet]['args'])
            modelDict[dataSet]['model'] = modelClass.model(args=modelDict[dataSet]['args'],
                                                           X_embedded_data=modelDict[dataSet]['WE'],
                                                           X_embedded_char=modelDict[dataSet]['CLWE'],
                                                           guideeInfo=None,
                                                           summery=modelDict[dataSet]['summery'],
                                                           scopename=dataSet)  # guideeInfo=None cuz in function we define
        dataNames = list()
        for dataSet in modelDict:
            modelDict[dataSet]['lossList'] = list()
            modelDict[dataSet]['f1ValList'] = list()
            modelDict[dataSet]['f1ValWOCRFList'] = list()
            modelDict[dataSet]['maxF1'] = 0.0
            modelDict[dataSet]['maxF1idx'] = 0
            modelDict[dataSet]['prevF1'] = 0.0
            modelDict[dataSet]['stop_counter'] = 0
            modelDict[dataSet]['early_stop'] = False
            modelDict[dataSet]['m_name'] = modelDict[dataSet]['args'].guidee_data

            dataNames.append(dataSet)

            try:
                os.mkdir('./modelSave/' + expName + '/' + modelDict[dataSet]['m_name'])
            except OSError as e:
                if e.errno == errno.EEXIST:  # if file exists! Python2.7 doesn't support file exist exception so need to use this
                    print('./modelSave/' + expName + '/' + modelDict[dataSet][
                        'm_name'] + ' Directory exists! not created.')
                    suffix += 1
                else:
                    raise

            modelDict[dataSet]['runner'] = RunModel(model=modelDict[dataSet]['model'], args=modelDict[dataSet]['args'],
                                                    ID2wordVecIdx=ID2wordVecIdx, ID2char=ID2char,
                                                    expName=expName, m_name=modelDict[dataSet]['m_name'],
                                                    m_train=m_train, m_dev=m_dev, m_test=m_test)

        self.modelDict = modelDict
        return dataset_name


    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):  # <--- implemented PER class
        """
        Reads a dataset in preparation for train or test. Returns data in proper format for train or test.
        Args:
            file_dict: dictionary
                 {
                    "train": dict, {key="file description":value="file location"},
                    "dev" : dict, {key="file description":value="file location"},
                    "test" : dict, {key="file description":value="file location"},
                 }
            dataset_name: str
                Name of the dataset required for calling appropriate utils, converters
        Returns:
            data: data in arbitrary format for train or test.
        Raises:
            None
        """

        datasets = self.read_helper(file_dict, dataset_name, kwargs['lr_decay'], kwargs['epoch'])
        self.datasets = datasets

        return datasets


    def train_helper(self, dataNames):
        m_train = self.m_train
        m_dev = self.m_dev
        m_test = self.m_test

        with tf.Session(config=self.gpu_config) as sess:
            phase = 0
            random.seed(seedV)
            np.random.seed(seedV)
            tf.set_random_seed(seedV)
            _ = tf.Variable(initial_value='fake_variable')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=100000)
            loader = tf.train.Saver(max_to_keep=100000)
            for epoch_idx in range(self.args.epoch * len(dataNames)):
                dataSet = dataNames[epoch_idx % len(dataNames)]
                if epoch_idx % len(dataNames) == 0:
                    if self.args.pretrained != 0:
                        phase += 1
                    print("[%d phase]" % (phase))
                m_name = self.modelDict[dataSet]['m_name']
                if self.modelDict[dataSet]['early_stop']:
                    continue
                if self.modelDict[dataSet]['args'].tensorboard:
                    tbWriter = tf.summary.FileWriter('./modelSave/' + self.expName + '/' + m_name + '/train', sess.graph)
                else:
                    tbWriter = None

                print('====' + m_name.upper() + "_MODEL Training=====")
                startTime = time.time()
                batch_idx = random.sample(range(0, len(self.modelDict[dataSet]['runner'].m_batchgroup[m_train])),
                                          len(self.modelDict[dataSet]['runner'].m_batchgroup[m_train]))

                if self.args.pretrained == 0:
                    intOuts = None
                    early_stops = [24, 30, 30, 30, 25, 25]

                if self.args.pretrained != 0:
                    intOuts = None
                    early_stops = [5, 16, 23, 30, 10, 16]

                self.early_stops = early_stops
                if ((epoch_idx / len(dataNames)) == 0) and (self.args.pretrained != 0):
                    intOuts = dict()
                    intOuts[m_train] = list()
                    intOuts[m_dev] = list()
                    intOuts[m_test] = list()
                    for d_sub in self.modelDict:
                        if d_sub == dataSet:
                            continue
                        else:
                            loadpath = './modelSave/' + str(self.args.pretrained) + '/' + d_sub + '/'
                            loader.restore(sess, tf.train.latest_checkpoint(loadpath))

                            intOuts[m_train].append(
                                self.modelDict[d_sub]['runner'].info1epoch(m_train, self.modelDict[dataSet]['runner'], sess))
                            intOuts[m_dev].append(
                                self.modelDict[d_sub]['runner'].info1epoch(m_dev, self.modelDict[dataSet]['runner'], sess))
                            intOuts[m_test].append(
                                self.modelDict[d_sub]['runner'].info1epoch(m_test, self.modelDict[dataSet]['runner'], sess))

                    loadpath = './modelSave/' + str(self.args.pretrained) + '/' + dataSet + '/'
                    loader.restore(sess, tf.train.latest_checkpoint(loadpath))

                elif ((epoch_idx / len(dataNames)) != 0):
                    if self.args.pretrained != 0:
                        intOuts = dict()
                        intOuts[m_train] = list()
                        intOuts[m_dev] = list()
                        intOuts[m_test] = list()
                        for d_sub in self.modelDict:
                            if d_sub == dataSet:
                                continue
                            else:
                                loadpath = './modelSave/' + expName + '/' + d_sub + '/'
                                loader.restore(sess, tf.train.latest_checkpoint(loadpath))
                                intOuts[m_train].append(
                                    self.modelDict[d_sub]['runner'].info1epoch(m_train, self.modelDict[dataSet]['runner'], sess))
                                intOuts[m_dev].append(
                                    self.modelDict[d_sub]['runner'].info1epoch(m_dev, self.modelDict[dataSet]['runner'], sess))
                                intOuts[m_test].append(
                                    self.modelDict[d_sub]['runner'].info1epoch(m_test, self.modelDict[dataSet]['runner'], sess))

                    loadpath = './modelSave/' + expName + '/' + dataSet + '/'
                    loader.restore(sess, tf.train.latest_checkpoint(loadpath))

                (l, sl, tra, trsPara) = self.modelDict[dataSet]['runner'].train1epoch(
                    sess, batch_idx, infoInput=intOuts, tbWriter=tbWriter)

                self.trsPara = trsPara
                print("== Epoch:%4d == | train time : %d Min | \n train loss: %.6f" % (
                epoch_idx, (time.time() - startTime) / 60, l))
                self.modelDict[dataSet]['lossList'].append(l)

            self.sess = sess

    def train(self, data, *args, **kwargs):  # <--- implemented PER class
        """
        Trains a model on the given input data
        Args:
            data: iterable of arbitrary format. represents the data instances and features and labels you use to train your model.
        Returns:
            ret: None. Trained model stored internally to class instance state.
        Raises:
            None
        """

        self.train_helper(data)

    def predict_helper(dataNames):
        modelDict = self.modelDict
        args = self.args

        m_train = self.m_train
        m_dev = self.m_dev
        m_test = self.m_test

        sess = self.sess

        for epoch_idx in range(self.args.epoch * len(dataNames)):
            dataSet = dataNames[epoch_idx % len(dataNames)]
            trsPara = self.trsPara
            m_name = self.modelDict[dataSet]['m_name']
            early_stops = self.early_stops
            (t_predictionResult, t_prfValResult, t_prfValWOCRFResult,
             test_x, test_ans, test_len) = modelDict[dataSet]['runner'].dev1epoch(m_test, trsPara, sess,
                                                                                  infoInput=intOuts, epoch=epoch_idx)

            modelDict[dataSet]['f1ValList'].append(t_prfValResult[2])
            saver.save(sess, './modelSave/' + self.expName + '/' + m_name + '/modelSaved')
            pickle.dump(trsPara, open('./modelSave/' + self.expName + '/' + m_name + '/trs_param.pickle', 'wb'))

            if ((epoch_idx / len(dataNames)) == early_stops[epoch_idx % len(dataNames)]):
                modelDict[dataSet]['early_stop'] = True
                modelDict[dataSet]['maxF1'] = t_prfValResult[2]
                modelDict[dataSet]['stop_counter'] = 0
                modelDict[dataSet]['maxF1idx'] = epoch_idx
                modelDict[dataSet]['trs_param'] = trsPara
                modelDict[dataSet]['maxF1_x'] = test_x[:]
                modelDict[dataSet]['maxF1_ans'] = test_ans[:]
                modelDict[dataSet]['maxF1_len'] = test_len[:]
                pickle.dump(modelDict[dataSet]['maxF1idx'],
                            open('./modelSave/' + self.expName + '/' + dataSet + '/maxF1idx.pickle', 'wb'))
                if args.pretrained != 0:
                    pickle.dump(intOuts[m_test],
                                open('./modelSave/' + self.expName + '/' + dataSet + '/bestInouts.pickle', 'wb'))

            for didx, dname in enumerate(dataNames):
                if not modelDict[dname]['early_stop']:
                    esFlag = False
                    break
                if modelDict[dname]['early_stop'] and didx == len(dataNames) - 1:
                    esFlag = True

            if esFlag:
                break


    def predict(self, data, *args, **kwargs):  # <--- implemented PER class WITH requirement on OUTPUT format!
        """
        Predicts on the given input data. Assumes model has been trained with train()
        Args:
            data: iterable of arbitrary format. represents the data instances and features you use to make predictions
                Note that prediction requires trained model. Precondition that class instance already stores trained model
                information.
        Returns:
            predictions: [tuple,...], i.e. list of tuples.
                Each tuple is (start index, span, mention text, mention type)
                Where:
                 - start index: int, the index of the first character of the mention span. None if not applicable.
                 - span: int, the length of the mention. None if not applicable.
                 - mention text: str, the actual text that was identified as a named entity. Required.
                 - mention type: str, the entity/mention type. None if not applicable.
                 NOTE: len(predictions) should equal len(data) AND the ordering should not change [important for
                     evalutation. See note in evaluate() about parallel arrays.]
        Raises:
            None
        """
        self.predict_helper(data)


    def evaluate_helper(self):
        modelDict = self.modelDict

        # Get test result for each model
        for dataSet in modelDict:
            m_name = modelDict[dataSet]['args'].guidee_data
            print('====' + m_name.upper() + "_MODEL Test=====")
            with tf.Session(config=self.gpu_config) as sess:
                random.seed(seedV)
                np.random.seed(seedV)
                tf.set_random_seed(seedV)
                sess.run(tf.global_variables_initializer())
                loader = tf.train.Saver(max_to_keep=10000)
                loadpath = './modelSave/' + self.expName + '/' + m_name + '/'

                if self.args.pretrained != 0:
                    intOuts = dict()
                    intOuts[self.m_test] = pickle.load(open(loadpath + 'bestInouts.pickle', 'rb'))
                else:
                    intOuts = None

                trsPara = pickle.load(open(loadpath + 'trs_param.pickle', 'rb'))
                loader.restore(sess, tf.train.latest_checkpoint(loadpath))

                if modelDict[dataSet]['args'].tensorboard:
                    tbWriter = tf.summary.FileWriter('test')
                else:
                    tbWriter = None

                (t_predictionResult, t_prfValResult, t_prfValWOCRFResult,
                 test_x, test_ans, test_len) = modelDict[dataSet]['runner'].dev1epoch(m_test, trsPara, sess,
                                                                                      infoInput=intOuts, epoch=None,
                                                                                      report=True)

        return t_prfValResult


    def evaluate(self, predictions, groundTruths, *args,
                 **kwargs):  # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
        """
        Calculates evaluation metrics on chosen benchmark dataset [Precision,Recall,F1, or others...]
        Args:
            predictions: [tuple,...], list of tuples [same format as output from predict]
            groundTruths: [tuple,...], list of tuples representing ground truth.
        Returns:
            metrics: tuple with (p,r,f1). Each element is float.
        Raises:
            None
        """
        # pseudo-implementation
        # we have a set of predictions and a set of ground truth data.
        # calculate true positive, false positive, and false negative
        # calculate Precision = tp/(tp+fp)
        # calculate Recall = tp/(tp+fn)
        # calculate F1 using precision and recall

        # return (precision, recall, f1)

        self.evaluate_helper()


    def save_model(self, file):
        """
        :param file: Where to save the model - Optional function
        :return:
        """
        pass


    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """
        pass



def main():
    collaboNet_obj = CollaboNet()

    file_dict = {'train':'train_dev', 'dev':'dev', 'test':'test'}
    dateset_name = "BC2GM"
    datasets

    # Training
    collaboNet_obj.train(datasets)

    # Predictions
    collaboNet_obj.predict(collaboNet_obj.modelDict)

    # Evaluations
    evaluations = collaboNet_obj.evaluate(None, None)

    precision, recall, F1 = evaluations[0], evaluations[1], evaluations[2]

    print ("Precision :", precision)
    print ("Recall :", recall)
    print ("F1 score :", F1)

    return

if __name__ == '__main__':
    main()



