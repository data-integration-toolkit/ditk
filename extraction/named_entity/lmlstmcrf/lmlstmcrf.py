from extraction.named_entity.ner import Ner
from extraction.named_entity.lmlstmcrf.hparams import hparams as hp
from extraction.named_entity.lmlstmcrf.model.utils import *
from extraction.named_entity.lmlstmcrf.model.lm_lstm_crf import LM_LSTM_CRF
from extraction.named_entity.lmlstmcrf.model.crf import CRFRepack_WC, CRFLoss_vb
from extraction.named_entity.lmlstmcrf.model.evaluator import eval_wc
from extraction.named_entity.lmlstmcrf.model.predictor import predict_wc

from tqdm import tqdm

import os, sys, time
import functools
import torch
import torch.optim as optim
import torch.nn as nn

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Lmlstmcrf(Ner):

    def __init__(self):
        self.state = None
        if hp.gpu >= 0:
            torch.cuda.set_device(hp.gpu)

        print('setting:')
        print(hp)

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
        # IMPLEMENT CONVERSION. STRICT OUTPUT FORMAT REQUIRED.
        
        # return ground_truth

    def read_dataset(self, file_dict, dataset_name=None, *args, **kwargs):
        """
        :param file_dict: dictionary
                    {
                        "train": "location_of_train",
                        "test": "location_of_test",
                        "dev": "location_of_dev",
                    }
        :param args:
        :param kwargs:
        :return: dictionary of iterables
                    Format:
                    {
                        "train":[
                                    [ Line 1 tokenized],
                                    [Line 2 tokenized],
                                    ...
                                    [Line n tokenized]
                                ],
                        "test": same as train,
                        "dev": same as train
                    }
        """
        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()
                for i, line in enumerate(raw_data):
                    if len(line.strip()) > 0:
                        raw_data[i] = line.strip().split()
                    else:
                        raw_data[i] = list(line)
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        return data

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
        # load corpus
        print('loading corpus')
        lines = data["train"]
        test_lines = data["test"]
        dev_lines = data["dev"]

        dev_features, dev_labels = read_data(dev_lines)
        test_features, test_labels = read_data(test_lines)

        if hp.load_check_point:
            if os.path.isfile(hp.checkpoint):
                print("loading checkpoint: '{}'".format(hp.checkpoint))
                checkpoint_file = torch.load(hp.checkpoint, map_location='cpu')
                args.start_epoch = checkpoint_file['epoch']
                f_map = checkpoint_file['f_map']
                l_map = checkpoint_file['l_map']
                c_map = checkpoint_file['c_map']
                in_doc_words = checkpoint_file['in_doc_words']
                train_features, train_labels = read_data(lines)
            else:
                print("no checkpoint found at: '{}'".format(hp.checkpoint))
        else:
            print('constructing coding table')
            # converting format
            train_features, train_labels, f_map, l_map, c_map = generate_corpus_char_from_data(lines, \
                if_shrink_c_feature=True, c_thresholds=hp.mini_count, if_shrink_w_feature=False)
            
            f_set = {v for v in f_map}
            f_map = shrink_features(f_map, train_features, hp.mini_count)

            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features), f_set)
            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features), dt_f_set)
            print("feature size: '{}'".format(len(f_map)))
            print('loading embedding')
            if hp.fine_tune:  # which means does not do fine-tune
                f_map = {'<eof>': 0}
            f_map, embedding_tensor, in_doc_words = load_embedding_wlm(hp.emb_file, ' ', f_map, dt_f_set, hp.caseless, \
                hp.unk, hp.word_dim, shrink_to_corpus=hp.shrink_embedding)
            print("embedding size: '{}'".format(len(f_map)))

            l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels))
            l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels), l_set)
            for label in l_set:
                if label not in l_map:
                    l_map[label] = len(l_map)

        print('constructing dataset')
        # construct dataset
        dataset, forw_corp, back_corp = construct_bucket_mean_vb_wc(train_features, train_labels, l_map,\
             c_map, f_map, hp.caseless)
        dev_dataset, forw_dev, back_dev = construct_bucket_mean_vb_wc(dev_features, dev_labels, l_map, c_map, \
            f_map, hp.caseless)
        test_dataset, forw_test, back_test = construct_bucket_mean_vb_wc(test_features, test_labels, l_map, c_map, \
            f_map, hp.caseless)
        
        dataset_loader = [torch.utils.data.DataLoader(tup, hp.batch_size, shuffle=True, drop_last=False) for tup in dataset]
        dev_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset]
        test_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset]

        # build model
        print('building model')
        self.ner_model = LM_LSTM_CRF(len(l_map), len(c_map), hp.char_dim, hp.char_hidden, hp.char_layers, hp.word_dim,\
             hp.word_hidden, hp.word_layers, len(f_map), hp.drop_out, large_CRF=hp.small_crf, if_highway=hp.high_way, \
                 in_doc_words=in_doc_words, highway_layers = hp.highway_layers)

        if hp.load_check_point:
            self.ner_model.load_state_dict(checkpoint_file['state_dict'])
        else:
            self.ner_model.load_pretrained_word_embedding(embedding_tensor)

        if hp.update == 'sgd':
            optimizer = optim.SGD(self.ner_model.parameters(), lr=hp.lr, momentum=hp.momentum)
        elif hp.update == 'adam':
            optimizer = optim.Adam(self.ner_model.parameters(), lr=hp.lr)

        if hp.load_check_point and hp.load_opt:
            optimizer.load_state_dict(checkpoint_file['optimizer'])

        crit_lm = nn.CrossEntropyLoss()
        crit_ner = CRFLoss_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if hp.gpu >= 0:
            if_cuda = True
            print('device: ' + str(hp.gpu))
            torch.cuda.set_device(hp.gpu)
            crit_ner.cuda()
            crit_lm.cuda()
            self.ner_model.cuda()
            packer = CRFRepack_WC(len(l_map), True)
        else:
            if_cuda = False
            packer = CRFRepack_WC(len(l_map), False)
        
        tot_length = sum(map(lambda t: len(t), dataset_loader))

        best_f1 = float('-inf')
        best_acc = float('-inf')
        self.track_list = list()
        start_time = time.time()
        epoch_list = range(hp.start_epoch, hp.start_epoch + hp.epoch)
        patience_count = 0

        evaluator = eval_wc(packer, l_map, hp.eva_matrix)

        for epoch_idx, hp.start_epoch in enumerate(epoch_list):
            epoch_loss = 0
            self.ner_model.train()
            for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v in tqdm(
                    itertools.chain.from_iterable(dataset_loader), mininterval=2,
                    desc=' - Tot it %d (epoch %d)' % (tot_length, hp.start_epoch), leave=False, file=sys.stdout):
                f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v)
                self.ner_model.zero_grad()
                scores = self.ner_model(f_f, f_p, b_f, b_p, w_f)
                loss = crit_ner(scores, tg_v, mask_v)
                epoch_loss += to_scalar(loss)
                if hp.co_train:
                    cf_p = f_p[0:-1, :].contiguous()
                    cb_p = b_p[1:, :].contiguous()
                    cf_y = w_f[1:, :].contiguous()
                    cb_y = w_f[0:-1, :].contiguous()
                    cfs, _ = self.ner_model.word_pre_train_forward(f_f, cf_p)
                    loss = loss + hp.lambda0 * crit_lm(cfs, cf_y.view(-1))
                    cbs, _ = self.ner_model.word_pre_train_backward(b_f, cb_p)
                    loss = loss + hp.lambda0 * crit_lm(cbs, cb_y.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.ner_model.parameters(), hp.clip_grad)
                optimizer.step()
            epoch_loss /= tot_length

            # update lr
            if hp.update == 'sgd':
                adjust_learning_rate(optimizer, hp.lr / (1 + (hp.start_epoch + 1) * hp.lr_decay))

            # eval & save check_point

            dev_result = evaluator.calc_score(self.ner_model, dev_dataset_loader)
            for label, (dev_f1, dev_pre, dev_rec, dev_acc, msg) in dev_result.items():
                print('DEV : %s : dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f | %s\n' % (label, dev_f1, dev_rec, dev_pre, dev_acc, msg))
            (dev_f1, dev_pre, dev_rec, dev_acc, msg) = dev_result['total']
            
            self.track_list.append(dev_result)

            if dev_f1 > best_f1:
                patience_count = 0
                best_f1 = dev_f1

                test_result = evaluator.calc_score(self.ner_model, test_dataset_loader)
                for label, (test_f1, test_pre, test_rec, test_acc, msg) in test_result.items():
                    print('TEST : %s : test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f | %s\n' % (label, test_f1, test_rec, test_pre, test_acc, msg))
                (test_f1, test_rec, test_pre, test_acc, msg) = test_result['total']

                print(
                    '(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f, F1 on test = %.4f, acc on test= %.4f), saving...' %
                    (epoch_loss,
                    hp.start_epoch,
                    dev_f1,
                    dev_acc,
                    test_f1,
                    test_acc))

                try:
                    self.state = {
                        'epoch': hp.start_epoch,
                        'state_dict': self.ner_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'l_map': l_map,
                        'c_map': c_map,
                        'in_doc_words': in_doc_words
                    }
                    self.save_model(hp.checkpoint_dir + 'cwlm_lstm_crf_' + hp.start_epoch)
                except Exception as inst:
                    print(inst)

            else:
                patience_count += 1
                print('(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f)' %
                      (epoch_loss,
                       hp.start_epoch,
                       dev_f1,
                       dev_acc))

            print('epoch: ' + str(hp.start_epoch) + '\t in ' + str(hp.epoch) + ' take: ' + str(
                time.time() - start_time) + ' s')

            if patience_count >= hp.patience and hp.start_epoch >= hp.least_iters:
                break

        #print best
        eprint(hp.checkpoint_dir + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc))
        self.state = {
            'epoch': hp.start_epoch,
            'state_dict': self.ner_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'f_map': f_map,
            'l_map': l_map,
            'c_map': c_map,
            'in_doc_words': in_doc_words
        }

        self.save_model(hp.checkpoint_dir + 'cwlm_lstm_crf_' + hp.start_epoch)
        return self.track_list

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
        # IMPLEMENT PREDICTION. STRICT OUTPUT FORMAT REQUIRED.
        lines = data["test"]

        # converting format
        features = read_features_from_data(lines)

        if hp.load_check_point and os.path.isfile(hp.checkpoint):
            checkpoint_file = torch.load(hp.checkpoint, map_location=lambda storage, loc: storage)
            f_map = checkpoint_file['f_map']
            l_map = checkpoint_file['l_map']
            c_map = checkpoint_file['c_map']
            in_doc_words = checkpoint_file['in_doc_words']
            if self.state == None:
                self.ner_model = LM_LSTM_CRF(len(l_map), len(c_map), hp.char_dim, hp.char_hidden, hp.char_layers, \
                    hp.word_dim, hp.word_hidden, hp.word_layers, len(f_map), hp.drop_out, large_CRF=hp.small_crf, \
                        if_highway=hp.high_way, in_doc_words=in_doc_words, highway_layers = hp.highway_layers)

            self.ner_model.load_state_dict(checkpoint_file['state_dict'])
            self.state = {
                'epoch': hp.start_epoch,
                'state_dict': self.ner_model.state_dict(),
                'f_map': f_map,
                'l_map': l_map,
                'c_map': c_map,
                'in_doc_words': in_doc_words
            }
        elif self.state != None:
            f_map = self.state['f_map']
            l_map = self.state['l_map']
            c_map = self.state['c_map']
            in_doc_words = self.state['in_doc_words']
        else:
            print("Train model first")
            return

        if_cuda = hp.gpu >= 0
        decode_label = (hp.decode_type == 'label')
        predictor = predict_wc(if_cuda, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'], l_map['<start>'], \
            decode_label, hp.batch_size, hp.caseless)

        print('annotating')
        predictions = predictor.output_predictions(self.ner_model, features)
        return predictions

    def evaluate(self, predictions, groundTruths, *args, **kwargs):  # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
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
        lines = groundTruths["test"]

        f_map = self.state['f_map']
        l_map = self.state['l_map']
        c_map = self.state['c_map']

        if hp.gpu >= 0:
            if_cuda = True
            packer = CRFRepack_WC(len(l_map), True)
        else:
            if_cuda = False
            packer = CRFRepack_WC(len(l_map), False)

        test_features, test_labels = read_data(lines)

        # construct dataset
        test_dataset, forw_test, back_test = construct_bucket_mean_vb_wc(test_features, test_labels, l_map, c_map, f_map,\
             hp.caseless)
        
        test_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset]

        evaluator = eval_wc(packer, l_map, hp.eva_matrix)

        print('start')
        result = evaluator.calc_score(self.ner_model, test_dataset_loader)
        for label, (test_f1, test_pre, test_rec, test_acc, msg) in result.items():
            print(hp.checkpoint +' : %s : test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f | %s\n' \
                % (label, test_f1, test_rec, test_pre, test_acc, msg))
        return (result["total"][1], result["total"][2], result["total"][0])
        	
    def save_model(self, file):
        """
        :param file: Where to save the model - Optional function
        :return:
        """
        save_checkpoint(self.state, {'track_list': self.track_list }, file)

    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """
        checkpoint_file = torch.load(file, map_location=lambda storage, loc: storage)
        f_map = checkpoint_file['f_map']
        l_map = checkpoint_file['l_map']
        c_map = checkpoint_file['c_map']
        in_doc_words = checkpoint_file['in_doc_words']
        if self.state == None:
            self.ner_model = LM_LSTM_CRF(len(l_map), len(c_map), hp.char_dim, hp.char_hidden, hp.char_layers, \
                hp.word_dim, hp.word_hidden, hp.word_layers, len(f_map), hp.drop_out, large_CRF=hp.small_crf, \
                    if_highway=hp.high_way, in_doc_words=in_doc_words, highway_layers = hp.highway_layers)

        self.ner_model.load_state_dict(checkpoint_file['state_dict'])

        if hp.update == 'sgd':
            optimizer = optim.SGD(self.ner_model.parameters(), lr=hp.lr, momentum=hp.momentum)
        elif hp.update == 'adam':
            optimizer = optim.Adam(self.ner_model.parameters(), lr=hp.lr)

        if hp.load_opt:
            optimizer.load_state_dict(checkpoint_file['optimizer'])

        self.state = {
            'epoch': hp.start_epoch,
            'state_dict': self.ner_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'f_map': f_map,
            'l_map': l_map,
            'c_map': c_map,
            'in_doc_words': in_doc_words
        }

        hp.load_check_point = True
        hp.checkpoint = file
    
