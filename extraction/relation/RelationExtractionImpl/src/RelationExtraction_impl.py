import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from relation.relation_extraction import RelationExtraction
from relation.RelationExtractionImpl.src.preprocessing import *
from relation.RelationExtractionImpl.src.joint_lstm import *
from relation.RelationExtractionImpl.src.utils import *
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class DDIExtractionImpl(RelationExtraction):
    def read_dataset(self, Input_files,  **kwargs):
        """
        reads and preprocesses the dataset to bring to a format ready for training
        :param InputFile: filepath containing the input files
        :return:dict with filepaths to inputfiles.
        """
        standard_split=["train","eval","test"]

        data={}
        try:
            for split in standard_split:
                file = Input_files[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        return data
        pass

    def train(self, Input_Data):
        """
        trains the model using the preprocessed data files form read_dataset
        :param: Input_data. File dict containing paths to train,dev and test files.
                Input_data={"train":"file1.txt",
                            "dev":"file2.txt",
                            "test":"file2.txt"}
        :return: None. Trained model stored internally
        """

        standard_split=["train","dev","test"]
        try:
            for split in standard_split:
                labels=convertdata(Input_Data[split],split+"step1.txt")
                processInput(split+"step1.txt",split+".txt")
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")

        embSize = 100
        d1_emb_size = 10
        d2_emb_size = 10
        type_emb_size = 10
        numfilter = 200
        num_epochs = 4
        # N = 4
        check_point = [3]
        batch_size = 200
        reg_para = 0.001
        drop_out = 1.0
        ftrain="train.txt"
        fval="dev.txt"
        ftest="test.txt"
        wefile = "PubMed-w2v.bin" #this word2vec file is downloaded from http://evexdb.org/pmresources/vec-space-models/

        Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables = dataRead(ftrain)
        Tr_word_list, Tr_d1_list, Tr_d2_list, Tr_type_list = makeFeatures(Tr_sent_contents, Tr_entity1_list,
                                                                          Tr_entity2_list)

        V_sent_contents, V_entity1_list, V_entity2_list, V_sent_lables = dataRead(fval)
        V_word_list, V_d1_list, V_d2_list, V_type_list = makeFeatures(V_sent_contents, V_entity1_list, V_entity2_list)

        Te_sent_contents, Te_entity1_list, Te_entity2_list, Te_sent_lables = dataRead(ftest)
        Te_word_list, Te_d1_list, Te_d2_list, Te_type_list = makeFeatures(Te_sent_contents, Te_entity1_list,
                                                                          Te_entity2_list)

        # print("train_size", len(Tr_word_list))
        # print("val_size", len(V_word_list))

        train_sent_lengths, val_sent_lengths, test_sent_lengths = findSentLengths(
            [Tr_word_list, V_word_list, Te_word_list])
        sentMax = max(train_sent_lengths + val_sent_lengths + test_sent_lengths)

        train_sent_lengths = np.array(train_sent_lengths, dtype='int32')
        val_sent_lengths = np.array(train_sent_lengths, dtype='int32')
        test_sent_lengths = np.array(test_sent_lengths, dtype='int32')

        # label_dict = {'false': 0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
        label_dict={'false':0}
        index=1
        for label in labels:
            if label == "false":
                continue
            label_dict[label]=index
            index +=1

        word_dict = makeWordList([Tr_word_list, V_word_list,Te_word_list])
        d1_dict = makeDistanceList([Tr_d1_list, V_d1_list,Te_d1_list])
        d2_dict = makeDistanceList([Tr_d2_list, V_d2_list,Te_d2_list])
        type_dict = makeDistanceList([Tr_type_list, V_type_list,Te_type_list])
        wv = readWordEmb(word_dict, wefile, embSize)
        W_train = mapWordToId(Tr_word_list, word_dict)
        d1_train = mapWordToId(Tr_d1_list, d1_dict)
        d2_train = mapWordToId(Tr_d2_list, d2_dict)
        T_train = mapWordToId(Tr_type_list, type_dict)

        Y_t = mapLabelToId(Tr_sent_lables, label_dict)
        Y_train = np.zeros((len(Y_t), len(label_dict)))
        for i in range(len(Y_t)):
            Y_train[i][Y_t[i]] = 1.0

        # Mapping Validation
        W_val = mapWordToId(V_word_list, word_dict)
        d1_val = mapWordToId(V_d1_list, d1_dict)
        d2_val = mapWordToId(V_d2_list, d2_dict)
        T_val = mapWordToId(V_type_list, type_dict)

        Y_t = mapLabelToId(V_sent_lables, label_dict)
        Y_val = np.zeros((len(Y_t), len(label_dict)))
        for i in range(len(Y_t)):
            Y_val[i][Y_t[i]] = 1.0

        W_test = mapWordToId(Te_word_list, word_dict)
        d1_test = mapWordToId(Te_d1_list, d1_dict)
        d2_test = mapWordToId(Te_d2_list, d2_dict)
        T_test = mapWordToId(Te_type_list, type_dict)
        Y_t = mapLabelToId(Te_sent_lables, label_dict)
        Y_test = np.zeros((len(Y_t), len(label_dict)))
        for i in range(len(Y_t)):
            Y_test[i][Y_t[i]] = 1.0

        W_train, d1_train, d2_train, T_train, W_val, d1_val, d2_val, T_val = paddData(
    [W_train, d1_train, d2_train, T_train, W_val, d1_val, d2_val, T_val], sentMax)
        with open('train_test_rnn_data.pickle', 'wb') as handle:
            pickle.dump(W_train, handle)
            pickle.dump(d1_train, handle)
            pickle.dump(d2_train, handle)
            pickle.dump(T_train, handle)
            pickle.dump(Y_train, handle)
            pickle.dump(train_sent_lengths, handle)

            pickle.dump(W_val, handle)
            pickle.dump(d1_val, handle)
            pickle.dump(d2_val, handle)
            pickle.dump(T_val, handle)
            pickle.dump(Y_val, handle)
            pickle.dump(val_sent_lengths, handle)

            pickle.dump(W_test, handle)
            pickle.dump(d1_test, handle)
            pickle.dump(d2_test, handle)
            pickle.dump(T_test, handle)
            pickle.dump(Y_test, handle)
            pickle.dump(test_sent_lengths, handle)
            pickle.dump(Tr_word_list,handle)
            pickle.dump(V_word_list,handle)
            pickle.dump(wv, handle)
            pickle.dump(word_dict, handle)
            pickle.dump(d1_dict, handle)
            pickle.dump(d2_dict, handle)
            pickle.dump(type_dict, handle)
            pickle.dump(label_dict, handle)
            pickle.dump(sentMax, handle)
            pickle.dump(label_dict,handle)
        word_dict_size = len(word_dict)
        d1_dict_size = len(d1_dict)
        d2_dict_size = len(d2_dict)
        type_dict_size = len(type_dict)
        label_dict_size = len(label_dict)

        rev_word_dict = makeWordListReverst(word_dict)
        rev_label_dict = {0: 'false', 1: 'advise', 2: 'mechanism', 3: 'effect', 4: 'int'}
        rnn = RNN_Relation(label_dict_size,  # output layer size
                          word_dict_size,  # word embedding size
                          d1_dict_size,  # position embedding size
                          d2_dict_size,  # position embedding size
                          type_dict_size,  # type emb. size
                          sentMax,  # length of sentence
                          wv,  # word embedding
                          d1_emb_size=d1_emb_size,  # emb. length
                          d2_emb_size=d2_emb_size,
                          type_emb_size=type_emb_size,
                          num_filters=numfilter,  # number of hidden nodes in RNN
                          w_emb_size=embSize,  # dim. word emb
                          l2_reg_lambda=reg_para  # l2 reg
                          )
        train_len = len(W_train)

        loss_list = []

        test_res = []
        val_res = []

        fscore_val = []
        fscore_test = []
        num_batches_per_epoch = int(train_len / batch_size) + 1
        iii = 0  # Check point number
        for epoch in range(num_epochs):
            shuffle_indices = np.random.permutation(np.arange(train_len))
            W_tr = W_train[shuffle_indices]
            d1_tr = d1_train[shuffle_indices]
            d2_tr = d2_train[shuffle_indices]
            T_tr = T_train[shuffle_indices]
            Y_tr = Y_train[shuffle_indices]
            S_tr = train_sent_lengths[shuffle_indices]
            loss_epoch = 0.0
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, train_len)
                loss = rnn.train_step(W_tr[start_index:end_index], S_tr[start_index:end_index],
                                      d1_tr[start_index:end_index],
                                      d2_tr[start_index:end_index], T_tr[start_index:end_index],
                                      Y_tr[start_index:end_index],
                                      drop_out)
                loss_epoch += loss

            # print(loss_epoch)
            loss_list.append(round(loss_epoch, 5))

            # if (epoch%N) == 0:

            if epoch in check_point:
                iii += 1

                saver = tf.train.Saver()
                path = saver.save(rnn.sess, 'model_' + str(iii) + '.ckpt')

                # Validation
                y_pred_val, acc = test_step(rnn,W_val, val_sent_lengths, d1_val, d2_val, T_val, Y_val)
                y_true_val = np.argmax(Y_val, 1)

                #		print 'y_true_val', np.shape(y_true_val)
                #		print 'y_pred_val', np.shape(y_pred_val)

                fscore_val.append(f1_score(y_true_val, y_pred_val, [1, 2, 3, 4], average='micro'))
                val_res.append([y_true_val, y_pred_val])
        save_model(rnn)
        pass

    def predict(self, test_data=None, *args):
        """
        :param None. test data is taken and processed during train itself.
        :return: relation prediction [sentence, predicted relation label, true relation label]
        """
        ftest = "test.txt"
        embSize = 100
        d1_emb_size = 10
        d2_emb_size = 10
        type_emb_size = 10
        numfilter = 200
        reg_para = 0.001

        test_res = []
        fscore_test = []

        Te_sent_contents, Te_entity1_list, Te_entity2_list, Te_sent_lables = dataRead(ftest)
        # for i in range(5000,len(Te_sent_contents)):
        #     print(Te_sent_contents[i])
        Te_word_list, Te_d1_list, Te_d2_list, Te_type_list = makeFeatures(Te_sent_contents, Te_entity1_list,
                                                                          Te_entity2_list)
        test_sent_lengths = findSentLengths([Te_word_list])
        test_sent_lengths = np.array(test_sent_lengths[0], dtype='int32')

        with open('train_test_rnn_data.pickle', 'rb') as handle:
            W_train = pickle.load(handle)
            d1_train = pickle.load(handle)
            d2_train = pickle.load(handle)
            T_train = pickle.load(handle)
            Y_train = pickle.load(handle)
            train_sent_lengths = pickle.load(handle)

            W_val = pickle.load(handle)
            d1_val = pickle.load(handle)
            d2_val = pickle.load(handle)
            T_val = pickle.load(handle)
            Y_val = pickle.load(handle)
            val_sent_lengths = pickle.load(handle)

            W_test = pickle.load(handle)
            d1_test = pickle.load(handle)
            d2_test = pickle.load(handle)
            T_test = pickle.load(handle)
            Y_test = pickle.load(handle)
            test_sent_lengths = pickle.load(handle)
            Tr_word_list=pickle.load(handle)
            V_word_list=pickle.load(handle)
            wv = pickle.load(handle)
            word_dict = pickle.load(handle)
            d1_dict = pickle.load(handle)
            d2_dict = pickle.load(handle)
            type_dict = pickle.load(handle)
            label_dict = pickle.load(handle)
            sentMax = pickle.load(handle)
            label_dict=pickle.load(handle)


        W_train, d1_train, d2_train, T_train, W_val, d1_val, d2_val, T_val, W_test, d1_test, d2_test, T_test = paddData(
            [W_train, d1_train, d2_train, T_train, W_val, d1_val, d2_val, T_val, W_test, d1_test, d2_test, T_test],
            sentMax)

        rnn=restore_model()
        y_pred_test, acc = test_step(rnn,W_test, test_sent_lengths, d1_test, d2_test, T_test, Y_test)
        y_true_test = np.argmax(Y_test, 1)
        # true_values=y_true_test.tolist()

        fsent=open("relation_extraction_test_output.txt",'w')
        # rev_label_dict = {0: 'false', 1: 'advise', 2: 'mechanism', 3: 'effect', 4: 'int'}
        rev_label_dict={}
        for label in label_dict:
            rev_label_dict[label_dict[label]]=label
        rev_word_dict = makeWordListReverst(word_dict)
        sentences=[]

        for sent, slen,e1,e2, y_t, y_p in zip(W_test, test_sent_lengths,Te_d1_list,Te_d2_list,y_true_test, y_pred_test):
            row=[]
            sent_l = [str(rev_word_dict[sent[kk]]) for kk in range(slen)]
            s = ' '.join(sent_l)
            sentences.append(s)
            fsent.write(s + '\t')
            fsent.write("null" + '\t' + "null" + '\t')
            fsent.write(rev_label_dict[y_p]+'\t'+rev_label_dict[y_t])
            fsent.write('\n')
        fsent.close()
        result=[]
        return [sentences,y_pred_test,y_true_test]


    def evaluate(self, y_pred, y_true):
        """

        :param predictions: predictions obtained from predict()
        :param true_values: ground truth obtained from dataset, returned by predict()
        :return: tuple with precision,recall,f1 score
        """

        p = precision_score(y_true, y_pred, [1, 2, 3, 4], average='micro')
        r=recall_score(y_true, y_pred, [1, 2, 3, 4], average='micro')
        f1=f1_score(y_true, y_pred, [1, 2, 3, 4], average='micro')
        return [p,r,f1]


    def tokenize(self, input_data, ngram_size=None):
        """
        method in parent class that is not required by my implementation
        :param input_data: input data
        :param ngram_size: None
        :return: None.
        """
        pass

    def data_preprocess(self, input_data):
        """
        not using this method of parent class as the preprocess method used in my implementation is not exposed to
        the user and is not tokenizer
        :param input_data: input data
        :return: None
        """
        pass
    def main(self,inputfile):
        obj=DDIExtractionImpl()
        input = {"train": inputfile,
                 "dev": inputfile,
                 "test": inputfile}
        obj.train(input)
        sent_pred = obj.predict()
        scores = obj.evaluate(sent_pred[1], sent_pred[2])
        return "relation_extraction_test_output.txt"

# if __name__ == '__main__':
#     obj = DDIExtractionImpl()
#     obj.main("relation_extraction_test_input")

