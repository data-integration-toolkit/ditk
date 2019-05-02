from collections import OrderedDict

import model, utils
from model import NERModel
import preprocess, postprocess
import keras.backend as K

import time
import os
import numpy as np
import cPickle as pkl
import sys

from sklearn.metrics import recall_score, precision_score, f1_score

models_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")
eval_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "evaluation")
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

class byteNER():
    class OptionParser():
        class Options():
            def __init__(self, opt_dict):
                for key, value in opt_dict.iteritems():
                    setattr(self,key,value)
        def __init__(self):
            self.opt_dict = {}
            
        def add_option(self, flag, default=None, help=""):
            self.opt_dict[flag[2:]] = default
        
        def parse_args(self, args):
            n_pair = len(args)/2
            for i in range(n_pair):
                self.opt_dict[args[i*2][2:]] = args[i*2+1]
            opts = self.Options(self.opt_dict)
            return [opts]

    def __init__(self):
        pass

    def convert_ground_truth(self, data, ground_truth_file = "examples/test"):  
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
        def firstElement(tup):
            return tup[0]

        pred_texts = []
        with open(ground_truth_file+".src") as f:
            for line in f:
                pred_texts.append(line.strip())

        true_text_labels = []
        with open(ground_truth_file+".tgt") as f:
            for line in f:
                raw_line = line.strip().split()
                assert len(raw_line) %3 == 0
                n_triple = len(raw_line) / 3
                single_text_label = []
                for i in range(n_triple):
                    single_text_label.append([int(raw_line[3*i][1:]), int(raw_line[3*i+1][1:]), raw_line[3*i+2]])
                    single_text_label.sort(key=firstElement)
                true_text_labels.append(single_text_label)

        output_txt = []
        return_result = []
        for text, true_label in zip(pred_texts, true_text_labels):
            text_tokens = text.split()
            single_text = []
            single_text_span = []
            char_count = 0
            for token in text_tokens:
                single_text.append([token, "O","O"])
                single_text_span.append([char_count, len(token)])
                char_count += len(token) + 1
            
            # Match true label
            # O(n) algorithm label each non-"O" token with tags.
            # It's complicated because the spans in the labels do not exact match with the position of the tokens.
            label_index = 0
            for i, token in enumerate(single_text):
                if label_index >= len(true_label):
                    break
                if single_text_span[i][0] >= true_label[label_index][0]:
                    shift = 0
                    while i+shift < len(single_text_span) and true_label[label_index][0]+true_label[label_index][1] > single_text_span[i+shift][0]:
                        single_text[i+shift][1] = true_label[label_index][2]
                        shift += 1
                    label_index += 1
            
            output_txt.append(single_text)
            
            for i,(single_piece, span) in enumerate(zip(single_text, single_text_span)):
                return_result.append((span[0], span[1], single_piece[0], single_piece[1]))
        return return_result

    def common2byteNER(self, common_name, file_type):
        texts = []
        labels = []
        with open(common_name) as f:
            single_text = []
            single_label = []
            for line in f:
                raw_line = line.strip().split("\t")
                if len(raw_line)==0 or raw_line[0] == "":
                    texts.append(" ".join(single_text))
                    char_count = 0
                    start_span_tag = []
                    for (word_token, word_label) in zip(single_text, single_label):
                        if word_label != "O":
                            start_span_tag.extend(["S"+str(char_count), "L"+str(len(word_token)), word_label[2:]])
                        char_count += len(word_token) + 1
                    labels.append(start_span_tag)
                    single_text = []
                    single_label = []
                else:
                    single_text.append(raw_line[0])
                    single_label.append(raw_line[3])
        byteNERpath = "examples/"
        byteNERsrc = byteNERpath + file_type +".src"
        byteNERtgt = byteNERpath + file_type +".tgt"
        with open(byteNERsrc,"w") as srcf:
            with open(byteNERtgt,"w") as tgtf:
                for text, label in zip(texts, labels):
                    if text != "":
                        srcf.write(text+"\n")
                        tgtf.write(" ".join(label)+"\n") 

    def read_dataset(self, file_dict, dataset_name):
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
        for file_type, file_value in file_dict.iteritems():
            common_name = list(file_value.values())[0]
            self.common2byteNER(common_name, file_type)
        return None

    def train_model(self, model_params):
        """
        Train a specific model
        :param model_params:
        :return:
        """

        # Check evaluation script / folders
        if not os.path.isfile(eval_script):
            raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
        if not os.path.exists(eval_temp):
            os.makedirs(eval_temp)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        # Build model
        ner_model = model.NERModel(model_path=model_params['model_path'], parameters=model_params)
        m = ner_model.model

        model_params['ner_model'] = ner_model
        model_params['model'] = m
        file_ext = model_params['file_ext']

        # Train data
        start_time = time.time()
        print 'Start time:', start_time

        # Params
        train_data_file = model_params['train_data_file']
        dev_data_file = model_params['dev_data_file']
        max_chars_in_sample = str(model_params['max_chars_in_sample'])
        nb_workers = model_params['nb_workers']
        nb_epochs = model_params['nb_epochs']

        combined_ext = utils.generate_combined_feat_ext(model_params)
        if len(combined_ext) > 0:
            X_train_file = train_data_file + combined_ext
            X_dev_file = dev_data_file + combined_ext
        else:
            X_train_file = ''
            X_dev_file = ''

        y_train_file = train_data_file.replace('.bytedrop', '') + '.new.y.' + max_chars_in_sample + '.iobes' + file_ext
        y_dev_file = dev_data_file + '.new.y.' + max_chars_in_sample + '.iobes' + file_ext

        # Assign callbacks
        callbacks = self.assign_callbacks(model_params)

        # generate word data (second input to model)
        X_train_word_file = ''
        X_dev_word_file = ''
        batch_size = model_params['batch_size']  # note: all batches within train data must be the same size, all batches within dev data must be the same size
        if model_params['use_word_embeddings']:
            X_train_word_file = train_data_file.replace('.bytedrop', '') + '.word-coded.' + str(max_chars_in_sample) + '.pkl'
            X_dev_word_file = dev_data_file + '.word-coded.' + str(max_chars_in_sample) + '.pkl'
            num_train_steps = utils.calc_batch_steps_size(X_train_word_file,
                                                          batch_size)  # note: if batch sizes aren't equal, last batch will wrap around to first batch and shift future batches, but this is ok.
            num_dev_steps = utils.calc_batch_steps_size(X_dev_word_file,
                                                        batch_size)  # note: if batch sizes aren't equal, last batch will wrap around to first batch and shift future batches, but this is ok.
        elif model_params['use_bpe_embeddings']:
            X_train_word_file = train_data_file.replace('.bytedrop', '') + '.bpe-embed-coded-' + os.path.basename(model_params['bpe_codes_file']) + '.' + str(max_chars_in_sample) + '.pkl'
            X_dev_word_file = dev_data_file + '.bpe-embed-coded-' + os.path.basename(model_params['bpe_codes_file']) + '.' + str(max_chars_in_sample) + '.pkl'
            num_train_steps = utils.calc_batch_steps_size(X_train_word_file,
                                                          batch_size)  # note: if batch sizes aren't equal, last batch will wrap around to first batch and shift future batches, but this is ok.
            num_dev_steps = utils.calc_batch_steps_size(X_dev_word_file,
                                                        batch_size)  # note: if batch sizes aren't equal, last batch will wrap around to first batch and shift future batches, but this is ok.
        else:
            num_train_steps = utils.calc_batch_steps_size(X_train_file,
                                                          batch_size)  # note: if batch sizes aren't equal, last batch will wrap around to first batch and shift future batches, but this is ok.
            num_dev_steps = utils.calc_batch_steps_size(X_dev_file,
                                                        batch_size)  # note: if batch sizes aren't equal, last batch will wrap around to first batch and shift future batches, but this is ok.

        # generate byte/bpe data (first input to model)
        train_data_gen_obj = utils.GenBatchForFit(nb_workers)
        train_data_gen = train_data_gen_obj.thread_gen_batch_data_for_fit(X_train_file, y_train_file, model_params, X_train_word_file)
        dev_data_gen_obj = utils.GenBatchForFit(nb_workers)
        dev_data_gen = dev_data_gen_obj.thread_gen_batch_data_for_fit(X_dev_file, y_dev_file, model_params, X_dev_word_file)

        # if reloading model, first run evaluation on it
        if model_params['reload']:
            metrics = callbacks[0]
            metrics.evaluate(model_params['dev_data_file'], metrics.dev_X_data, metrics.dev_r_tags, metrics.dev_y_reals, model_params)
            metrics.evaluate(model_params['test_data_file'], metrics.test_X_data, metrics.test_r_tags, metrics.test_y_reals, model_params)

        # TRAIN
        m.fit_generator(train_data_gen, steps_per_epoch=num_train_steps, epochs=nb_epochs, validation_data=dev_data_gen, validation_steps=num_dev_steps, workers=nb_workers, callbacks=callbacks)

        print '---- %s epochs trained in %.4fs ----' % (model_params['nb_epochs'], time.time() - start_time)

    def assign_callbacks(self, model_params):
        """
        Callbacks for the model
        :param model_params:
        :return:
        """
        metrics = utils.MetricsCheckpoint(model_params['model_path'], model_params)
        callbacks = [metrics]
        return callbacks

    def train(self, data, nb_epochs=300):  
        """
        Trains a model on the given input data
        Args:
            data: iterable of arbitrary format. represents the data instances and features and labels you use to train your model.
        Returns:
            ret: None. Trained model stored internally to class instance state.
        Raises:
            None
        """
        args = ['--model_path', 'models/example.model', '--train_data_file', 'examples/train', '--dev_data_file', 'examples/dev', '--test_data_file', 'examples/test', '--nb_epochs', str(nb_epochs), '--batch_size', '10']

        ####### Code from original byteNER #####
        parser = self.OptionParser()
        parser.add_option('--model_path', help='Output file to save model to')
        parser.add_option('--input_format', default='st', help='Input format [iob|st]')
        parser.add_option('--space_token', default='<SPACE>', help='If input format is iob, then use space_token for spaces between bytes')
        parser.add_option('--train_data_file', help='Training data: if input_format == "st", then there are two input files: train_data_file.src and train_data_file.tgt')
        parser.add_option('--dev_data_file', help='Dev data: if input_format == "st", then there are two input files: dev_data_file.src and dev_data_file.tgt')
        parser.add_option('--test_data_file', help='Test data: if input_format == "st", then there are two input files: test_data_file.src and test_data_file.tgt')
        parser.add_option('--max_chars_in_sample', default=150, help='Max number of characters in a data sample')
        parser.add_option('--embedding_input_dim', default=256 + 1, help='Dimension of input vectors')
        parser.add_option('--embedding_output_dim', default=100, help='Dimension of output embedding vectors')
        parser.add_option('--embedding_max_len', default=150, help='Set length of input data (in byte characters)')
        parser.add_option('--cnn_filters', default=250, help='Number of filters in CNN output')
        parser.add_option('--cnn_kernel_size', default=7, help='Kernel size')
        parser.add_option('--cnn_padding', default='same', help='Type of border for CNN')
        parser.add_option('--cnn_act', default='relu', help='Activation fn for CNN')
        parser.add_option('--dense_final_act', default='softmax', help='Final activation fn in network')
        parser.add_option('--optimizer', default='adam', help='Optimizer')
        parser.add_option('--tag_scheme', default='iobes', help='IOBES or IOB2 tag scheme')
        parser.add_option('--batch_size', default=256, help='Number of samples to process in one batch')
        parser.add_option('--dropout', default=0.5, help='Fraction of input units to dropout')
        parser.add_option('--lstm_units', default=100, help=0)
        parser.add_option('--lstm_act', default='tanh', help='Activation fn for LSTM')
        parser.add_option('--num_byte_layers', default=20, help='Number of CNN layers in architecture')
        parser.add_option('--nb_workers', default=1, help='Number of threads or processes to use')
        parser.add_option('--nb_epochs', default=300, help='Number of epochs to train model for')
        parser.add_option('--residual', default=1, help='Whether to use residual connections')
        parser.add_option('--skip_residuals', default=0, help='Whether to add a residual connection every other conv layer')
        parser.add_option('--lr', default=0.00005, help='Learning rate of optimizer')
        parser.add_option('--use_bpe', default=0, help='Whether to use byte pair encodings')
        parser.add_option('--num_operations', default=50000, help="Number of merge operations for BPE algorithm")
        parser.add_option('--reload', default=0, help="Whether to reload a previously trained model")
        parser.add_option('--blstm_on_top', default=1, help="Whether to use a BLSTM layer on top of the CNNs")
        parser.add_option('--crf_on_top', default=1, help="Whether to use a CRF layer on top")
        parser.add_option('--use_word_embeddings', default=0, help="Whether to also use pretrained word embeddings as input")
        parser.add_option('--use_bytes', default=1, help="Whether to use byte embedding as input. Default is true.")
        parser.add_option('--word_embeddings_file', default='', help="Pretrained word embedding file. This needs to be set in order to use word embeddings.")
        parser.add_option('--word_embedding_dim', default=200, help="Dimension of pretrained word embeddings")
        parser.add_option('--bpe_codes_file', default='pmc_codes_file_ALL_50000', help="Pretrained BPE file")
        parser.add_option('--use_bpe_embeddings', default=1, help="Whether to use pretrained BPE embeddings as input. Cannot be used with word embeddings")
        parser.add_option('--bpe_embeddings_file', default='models/bpe2vec.csv', help="Pretrained BPE embedding file. This needs to be set in order to use BPE embeddings")
        parser.add_option('--bpe_embedding_dim', default=100, help="Dimension of pretrained BPE embeddings")
        parser.add_option('--byte_layer_for_embed', default=1, help="Whether to use embedding inputs as inputs for the CNN layers (or only BLSTM layers)")
        parser.add_option('--layer_for_bytes', default='cnn', help="stack of cnns or a blstm layer for bytes")
        parser.add_option('--temp_dir', default='evaluation/temp', help="Directory to write evaluation scores")
        parser.add_option('--drop_bytes', default=1, help="Whether to drop a fraction of bytes for each input")
        parser.add_option('--byte_drop_fraction', default=0.3, help="Fraction of byte input to drop")
        parser.add_option('--train_data_stride', default=75, help="Stride in number of bytes to shift window to get next training sample")
        parser.add_option('--trainable_bpe_embeddings', default=1, help="Whether BPE embeddings should be traininable")
        parser.add_option('--override_parameters', default=0, help="Whether to use specified parameters or parameters from saved model, if they exist")
        parser.add_option('--get_probs', default=0, help="Get normalized log likelihoods of each sample")
        parser.add_option('--get_vectors', default=0, help="Get output vectors of second-to-last layer in the network. Currently only tested with the CNN-BLSTM-CRF configuration")
        parser.add_option('--use_tokenization', default=0, help="Use tokenization features")
        parser.add_option('--repickle_data', default=0, help="Whether to re-process and pickle data even if the pickle file already exists")
        parser.add_option('--make_samples_unique', default=0, help="Clean training data so that the user-provided samples are all unique")
        opts = parser.parse_args(args)[0]

        # Parameters
        parameters = OrderedDict()
        parameters['override_parameters'] = int(opts.override_parameters) == 1
        parameters['model_path'] = opts.model_path
        parameters['reload'] = int(opts.reload) == 1
        parameters_path = parameters['model_path'] + '_parameters.pkl'

        if parameters['reload'] or parameters['override_parameters']:  # load parameters from file
            with open(parameters_path, 'rb') as f:
                parameters = pkl.load(f)
        else:  # set parameters
            parameters['max_chars_in_sample'] = int(opts.max_chars_in_sample)
            parameters['embedding_input_dim'] = int(opts.embedding_input_dim)
            parameters['embedding_output_dim'] = int(opts.embedding_output_dim)
            parameters['embedding_max_len'] = int(opts.embedding_max_len)
            parameters['cnn_filters'] = int(opts.cnn_filters)
            parameters['cnn_kernel_size'] = int(opts.cnn_kernel_size)
            parameters['cnn_padding'] = opts.cnn_padding
            parameters['cnn_act'] = opts.cnn_act
            parameters['dense_final_act'] = opts.dense_final_act
            parameters['optimizer'] = opts.optimizer
            parameters['tag_scheme'] = opts.tag_scheme
            parameters['dropout'] = float(opts.dropout)
            parameters['lstm_units'] = int(opts.lstm_units)
            parameters['lstm_act'] = opts.lstm_act
            parameters['num_byte_layers'] = int(opts.num_byte_layers)
            parameters['residual'] = int(opts.residual) == 1
            parameters['skip_residuals'] = int(opts.skip_residuals) == 1
            parameters['lr'] = float(opts.lr)
            parameters['use_bpe'] = int(opts.use_bpe) == 1
            parameters['num_operations'] = int(opts.num_operations)  # will be overridden if bpe_codes_file is set
            parameters['blstm_on_top'] = int(opts.blstm_on_top) == 1
            parameters['crf_on_top'] = int(opts.crf_on_top) == 1
            parameters['use_bytes'] = int(opts.use_bytes) == 1
            parameters['word_embeddings_file'] = opts.word_embeddings_file
            if parameters['word_embeddings_file']:
                parameters['use_word_embeddings'] = int(opts.use_word_embeddings) == 1
            else:
                parameters['use_word_embeddings'] = False
            parameters['word_embeddings_dim'] = int(opts.word_embedding_dim)
            parameters['bpe_codes_file'] = opts.bpe_codes_file
            if parameters['bpe_codes_file']:
                with open(parameters['bpe_codes_file'], 'rU') as f:
                    lines = f.readlines()
                num_operations = len(lines) - 2
                parameters['num_operations'] = num_operations
            parameters['bpe_embeddings_file'] = opts.bpe_embeddings_file
            if parameters['bpe_embeddings_file']:
                parameters['use_bpe_embeddings'] = int(opts.use_bpe_embeddings) == 1
            else:
                parameters['use_bpe_embeddings'] = False
            parameters['bpe_embeddings_dim'] = int(opts.bpe_embedding_dim)
            parameters['byte_layer_for_embed'] = int(opts.byte_layer_for_embed) == 1
            parameters['layer_for_bytes'] = opts.layer_for_bytes
            parameters['trainable_bpe_embeddings'] = int(opts.trainable_bpe_embeddings) == 1
            parameters['use_tokenization'] = int(opts.use_tokenization) == 1  # in IOBES format
            parameters['space_token'] = opts.space_token

        # set data files
        parameters['train_data_file'] = opts.train_data_file
        parameters['dev_data_file'] = opts.dev_data_file
        parameters['test_data_file'] = opts.test_data_file
        parameters['get_probs'] = int(opts.get_probs) == 1
        parameters['get_vectors'] = int(opts.get_vectors) == 1
        parameters['repickle_data'] = int(opts.repickle_data) == 1
        parameters['input_format'] = opts.input_format
        parameters['make_samples_unique'] = int(opts.make_samples_unique) == 1
        parameters['temp_dir'] = opts.temp_dir
        parameters['drop_bytes'] = int(opts.drop_bytes) == 1
        parameters['byte_drop_fraction'] = float(opts.byte_drop_fraction)
        parameters['train_data_stride'] = int(opts.train_data_stride)
        parameters['override_parameters'] = int(opts.override_parameters) == 1
        parameters['model_path'] = opts.model_path
        parameters['reload'] = int(opts.reload) == 1
        parameters['batch_size'] = int(opts.batch_size)
        parameters['nb_workers'] = int(opts.nb_workers)
        parameters['nb_epochs'] = int(opts.nb_epochs)

        # Convert user input to model input PKL format if PKL file doesn't already exist
        train_pkl_file = parameters['train_data_file'] + '.pkl'
        dev_pkl_file = parameters['dev_data_file'] + '.pkl'
        test_pkl_file = parameters['test_data_file'] + '.pkl'
        if parameters['repickle_data'] or parameters['make_samples_unique'] or not (os.path.isfile(train_pkl_file) and os.path.isfile(dev_pkl_file) and os.path.isfile(test_pkl_file)):
            train_src_file = parameters['train_data_file'] + '.src'
            train_tgt_file = parameters['train_data_file'] + '.tgt'
            dev_src_file = parameters['dev_data_file'] + '.src'
            dev_tgt_file = parameters['dev_data_file'] + '.tgt'
            test_src_file = parameters['test_data_file'] + '.src'
            test_tgt_file = parameters['test_data_file'] + '.tgt'

            if parameters['input_format'] == 'iob':  # convert to st format
                preprocess.write_user_byte_iob_input_to_src_tgt_input(parameters['train_data_file'], train_src_file, train_tgt_file, space_token=parameters['space_token'])
                preprocess.write_user_byte_iob_input_to_src_tgt_input(parameters['dev_data_file'], dev_src_file, dev_tgt_file, space_token=parameters['space_token'])
                preprocess.write_user_byte_iob_input_to_src_tgt_input(parameters['test_data_file'], test_src_file, test_tgt_file, space_token=parameters['space_token'])

            # train_pkl_file = parameters['train_data_file'] + '.pkl'
            # dev_pkl_file = parameters['dev_data_file'] + '.pkl'
            # test_pkl_file = parameters['test_data_file'] + '.pkl'

            if parameters['make_samples_unique']:  # unique training samples
                unique_train_file = utils.remove_duplicate_samples(parameters['train_data_file'])
                train_src_file = unique_train_file + '.src'
                train_tgt_file = unique_train_file + '.tgt'

            preprocess.write_user_input_to_model_input(train_src_file, train_tgt_file, train_pkl_file)
            preprocess.write_user_input_to_model_input(dev_src_file, dev_tgt_file, dev_pkl_file)
            preprocess.write_user_input_to_model_input(test_src_file, test_tgt_file, test_pkl_file)

        parameters['train_data_file'] = train_pkl_file
        parameters['dev_data_file'] = dev_pkl_file
        parameters['test_data_file'] = test_pkl_file

        # Load train, dev, and test data
        tags = utils.collect_tags(parameters['train_data_file'])
        parameters['tags'] = tags
        tag_to_num = utils.build_tag_to_num(tags)
        parameters['tag_to_num'] = tag_to_num
        char_to_num = utils.build_char_to_num()
        parameters['char_to_num'] = char_to_num
        if parameters['use_tokenization']:
            tok_char_to_num = utils.build_tok_char_to_num()
            parameters['tok_char_to_num'] = tok_char_to_num
        enc_dec_tag_to_num = utils.build_enc_dec_tag_to_num(parameters)
        parameters['enc_dec_tag_to_num'] = enc_dec_tag_to_num
        parameters['num_enc_dec_tags'] = len(enc_dec_tag_to_num)
        parameters['num_iobes_tags'] = len(tag_to_num)
        max_chars_in_sample = parameters['max_chars_in_sample']
        parameters['enc_dec_output_length'] = max_chars_in_sample

        print 'Parameters'
        for param_key in parameters:
            if 'vocab_dict' not in param_key:
                print '\t', param_key, ': ', parameters[param_key]

        parameters['file_ext'] = '.pkl'
        utils.load_data_stride_x_chars_enc_dec(parameters['train_data_file'], parameters, stride=parameters['train_data_stride'])
        utils.load_data_stride_x_chars_enc_dec(parameters['dev_data_file'], parameters, stride=max_chars_in_sample / 2)
        utils.load_data_stride_x_chars_enc_dec(parameters['test_data_file'], parameters, stride=max_chars_in_sample / 2)
        file_ext = parameters['file_ext']

        if parameters['drop_bytes']:
            char_to_num['<DROP>'] = len(char_to_num)
            utils.byte_dropout(parameters['train_data_file'], parameters)
            parameters['train_data_file'] += '.bytedrop'

        if parameters['use_bpe']:
            if parameters['bpe_codes_file']:
                codes_file = parameters['bpe_codes_file']
            else:
                codes_file = 'codes_file.' + str(parameters['num_operations'])
                if not os.path.isfile(codes_file):
                    # run bpe on train data
                    utils.gen_bpe_code_file(parameters['train_data_file'] + '.bpe.' + str(max_chars_in_sample) + '.pkl', parameters['num_operations'], codes_file)

            parameters['bpe_codes_file'] = codes_file

            # add code_file to char_to_num
            utils.add_bpe_to_vocab_dictionary(codes_file, char_to_num)

            # use codes to run bpe on data
            utils.load_bpe_data(parameters['train_data_file'].replace('.bytedrop', '') + '.bpe.' + str(max_chars_in_sample) + file_ext, char_to_num, codes_file,
                                parameters['train_data_file'].replace('.bytedrop', '') + '.bpe-coded-' + os.path.basename(parameters['bpe_codes_file']) + '.' + str(max_chars_in_sample) + file_ext, parameters)
            utils.load_bpe_data(parameters['dev_data_file'] + '.bpe.' + str(max_chars_in_sample) + file_ext, char_to_num, codes_file, parameters['dev_data_file'] + '.bpe-coded-' + os.path.basename(parameters['bpe_codes_file']) + '.' + str(max_chars_in_sample) + file_ext, parameters)
            utils.load_bpe_data(parameters['test_data_file'] + '.bpe.' + str(max_chars_in_sample) + file_ext, char_to_num, codes_file, parameters['test_data_file'] + '.bpe-coded-' + os.path.basename(parameters['bpe_codes_file']) + '.' + str(max_chars_in_sample) + file_ext, parameters)

        if parameters['use_tokenization']:
            utils.load_tok_data(
                parameters['train_data_file'].replace('.bytedrop', '') + '.bpe.' + str(max_chars_in_sample) + file_ext,
                tok_char_to_num,
                parameters['train_data_file'].replace('.bytedrop', '') + '.tok.' + str(max_chars_in_sample) + file_ext, parameters)
            utils.load_tok_data(
                parameters['dev_data_file'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
                tok_char_to_num,
                parameters['dev_data_file'] + '.tok.' + str(max_chars_in_sample) + file_ext,
                parameters)
            utils.load_tok_data(
                parameters['test_data_file'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
                tok_char_to_num,
                parameters['test_data_file'] + '.tok.' + str(max_chars_in_sample) + file_ext,
                parameters)

        if parameters['use_word_embeddings']:
            vocab_dict = OrderedDict()

            # add vocab to vocab_dict
            print 'Adding words...'
            utils.add_embeddings_to_vocab_dictionary(parameters['word_embeddings_file'], vocab_dict, parameters)
            print 'Done adding words'

            # add unknown, space, and pad embedding to vocab dictionary
            vocab_dict['<UNKNOWN>'] = (len(vocab_dict), np.random.rand(parameters['word_embeddings_dim']))
            vocab_dict['<SPACE>'] = (len(vocab_dict), np.random.rand(parameters['word_embeddings_dim']))

            # run word embedding features on dev and test data
            utils.load_word_embeddings_data(parameters['train_data_file'].replace('.bytedrop', '') + '.bpe.' + str(max_chars_in_sample) + '.pkl', vocab_dict,
                                parameters['train_data_file'].replace('.bytedrop', '') + '.word-coded.' + str(max_chars_in_sample) + '.pkl', parameters)
            utils.load_word_embeddings_data(parameters['dev_data_file'] + '.bpe.' + str(max_chars_in_sample) + '.pkl',
                                            vocab_dict,
                                            parameters['dev_data_file'] + '.word-coded.' + str(max_chars_in_sample) + '.pkl',
                                            parameters)
            utils.load_word_embeddings_data(parameters['test_data_file'] + '.bpe.' + str(max_chars_in_sample) + '.pkl',
                                            vocab_dict,
                                            parameters['test_data_file'] + '.word-coded.' + str(max_chars_in_sample) + '.pkl',
                                            parameters)

            parameters['word_vocab_dict'] = vocab_dict

        if parameters['use_bpe_embeddings']:
            if parameters['bpe_codes_file']:
                codes_file = parameters['bpe_codes_file']
            else:
                codes_file = 'codes_file.' + str(parameters['num_operations'])
                parameters['bpe_codes_file'] = codes_file

            vocab_dict = OrderedDict()

            # add vocab to vocab_dict
            print 'Adding BPE embeddings...'
            utils.add_embeddings_to_vocab_dictionary(parameters['bpe_embeddings_file'], vocab_dict, parameters)
            print 'Done adding BPE embeddings'

            # add unknown, space, and pad embedding to vocab dictionary
            vocab_dict['<UNKNOWN>'] = (len(vocab_dict), np.random.rand(parameters['bpe_embeddings_dim']))

            # run word embedding features on dev and test data
            utils.load_bpe_embeddings_data(parameters['train_data_file'].replace('.bytedrop', '') + '.bpe.' + str(max_chars_in_sample) + file_ext, vocab_dict, codes_file,
                                parameters['train_data_file'].replace('.bytedrop', '') + '.bpe-embed-coded-' + os.path.basename(codes_file) + '.' + str(max_chars_in_sample) + file_ext, parameters)
            utils.load_bpe_embeddings_data(parameters['dev_data_file'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
                                            vocab_dict,
                                            codes_file,
                                            parameters['dev_data_file'] + '.bpe-embed-coded-' + os.path.basename(codes_file) + '.' + str(max_chars_in_sample) + file_ext,
                                            parameters)
            utils.load_bpe_embeddings_data(parameters['test_data_file'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
                                            vocab_dict,
                                            codes_file,
                                            parameters['test_data_file'] + '.bpe-embed-coded-' + os.path.basename(codes_file) + '.'+ str(max_chars_in_sample) + file_ext,
                                            parameters)

            parameters['bpe_vocab_dict'] = vocab_dict

        # update embedding params
        parameters['embedding_input_dim'] = len(char_to_num)
        print 'Total embedding dimensions:', parameters['embedding_input_dim']

        # determine what will be the combined data file
        combined_data_ext = utils.generate_combined_feat_ext(parameters)

        # Combine data features as necessary
        utils.combine_data(parameters['train_data_file'], combined_data_ext, parameters)
        utils.combine_data(parameters['dev_data_file'], combined_data_ext, parameters)
        utils.combine_data(parameters['test_data_file'], combined_data_ext, parameters)

        # Save parameters
        if not parameters['reload'] and not parameters['override_parameters']:
            with open(parameters_path, 'wb') as p:
                pkl.dump(parameters, p)

        self.train_model(parameters)

        return None

    def load_data_and_predict(self, parameters):
        """
        Preprocess input data and make predictions.
        :param parameters:
        :return:
        """
        # Convert user input to model input PKL format
        pkl_file = parameters['input'] + '.pkl'
        preprocess.write_user_input_to_model_input(parameters['input'], '', pkl_file)
        parameters['input'] = pkl_file

        max_chars_in_sample = parameters['max_chars_in_sample']
        utils.load_data_stride_x_chars_enc_dec(parameters['input'], parameters, stride=max_chars_in_sample / 2)
        file_ext = parameters['file_ext']
        char_to_num = parameters['char_to_num']

        if parameters['use_bpe']:
            if parameters['bpe_codes_file']:
                codes_file = parameters['bpe_codes_file']
            else:
                print 'BPE codes file does not exist!'
                exit()
            utils.load_bpe_data(parameters['input'] + '.bpe.' + str(max_chars_in_sample) + file_ext, char_to_num, codes_file, parameters['input'] + '.bpe-coded-' + os.path.basename(parameters['bpe_codes_file']) + '.' + str(max_chars_in_sample) + file_ext, parameters)

        if parameters['use_tokenization']:
            tok_char_to_num = parameters['tok_char_to_num']
            utils.load_tok_data(
                parameters['input'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
                tok_char_to_num,
                parameters['input'] + '.tok.' + str(max_chars_in_sample) + file_ext, parameters)

        if parameters['use_word_embeddings']:
            vocab_dict = parameters['word_vocab_dict']
            # run word embedding features on dev and test data
            utils.load_word_embeddings_data(parameters['input'] + '.bpe.' + str(max_chars_in_sample) + '.pkl',
                                            vocab_dict,
                                            parameters['input'] + '.word-coded.' + str(max_chars_in_sample) + '.pkl',
                                            parameters)
        if parameters['use_bpe_embeddings']:
            if parameters['bpe_codes_file']:
                codes_file = parameters['bpe_codes_file']
            else:
                print 'BPE codes file does not exist!'
                exit()
            vocab_dict = parameters['bpe_vocab_dict']
            # run word embedding features on dev and test data
            utils.load_bpe_embeddings_data(parameters['input'] + '.bpe.' + str(max_chars_in_sample) + file_ext,
                                            vocab_dict,
                                            codes_file,
                                            parameters['input'] + '.bpe-embed-coded-' + os.path.basename(codes_file) + '.' + str(max_chars_in_sample) + file_ext,
                                            parameters)

        # determine what will be the combined data file
        combined_data_ext = utils.generate_combined_feat_ext(parameters)

        # Combine data features as necessary
        utils.combine_data(parameters['input'], combined_data_ext, parameters)

        # Make predictions
        metrics = utils.MetricsCheckpoint(parameters['model'], parameters)
        metrics.make_and_format_predictions(parameters['input'], metrics.test_X_data, parameters)

        # Output in specified format
        if parameters['output_format'] == 'iob':
            postprocess.ord_byte_iob_to_byte_iob(parameters['output'])
        else:  # st
            postprocess.byte_iob_to_src_tgt(parameters['output'], ordinal=True)


    def predict(self, data, input_file = 'examples/test', output_file = 'examples/pred'):  # <--- implemented PER class WITH requirement on OUTPUT format!
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
        args = ['--model', 'models/example.model', '--input', input_file+".src", '--output', output_file]
        # User parameters
        parser = self.OptionParser()
        parser.add_option(
            "--model", default="",
            help="Model location"
        )
        parser.add_option(
            "--input", default="",
            help="Input file, one sample per line"
        )
        parser.add_option(
            "--output", default="",
            help="Output file location"
        )
        parser.add_option(
            "--output_format", default="st",
            help="Whether to output predicted tokens in IOB format or src/tgt format. [iob|st]"
        )
        parser.add_option('--get_probs', default=0, help="Get normalized log likelihoods of each sample")
        parser.add_option('--get_vectors', default=0,
                          help="Get output vectors of second-to-last layer in the network. Currently only tested with the CNN-BLSTM-CRF configuration")
        opts = parser.parse_args(args)[0]

        # Check parameters validity
        assert opts.output_format in ["iob", "st"]
        assert os.path.isfile(opts.model)
        assert os.path.isfile(opts.model + "_parameters.pkl")  # need params file to reload model
        assert os.path.isfile(opts.input)

        # Add parameters
        parameters = {'reload': True, 'tag': True, 'repickle_data': True}

        # Load existing model
        print "Loading model..."
        model = NERModel(model_path=opts.model, parameters=parameters)
        parameters = model.parameters
        parameters['input'] = opts.input
        parameters['output'] = opts.output
        parameters['output_format'] = opts.output_format
        parameters['model'] = model.model
        parameters['get_probs'] = int(opts.get_probs) == 1
        parameters['get_vectors'] = int(opts.get_vectors) == 1

        print 'Tagging...'
        start = time.time()
        self.load_data_and_predict(parameters)
        print '---- lines tagged in %.4fs ----' % (time.time() - start)

        ############## End of original byteNER code #############

        def firstElement(tup):
            return tup[0]

        pred_texts = []
        with open(output_file+".src") as f:
            for line in f:
                pred_texts.append(line.strip())

        true_text_labels = []
        with open(input_file+".tgt") as f:
            for line in f:
                raw_line = line.strip().split()
                assert len(raw_line) %3 == 0
                n_triple = len(raw_line) / 3
                single_text_label = []
                for i in range(n_triple):
                    single_text_label.append([int(raw_line[3*i][1:]), int(raw_line[3*i+1][1:]), raw_line[3*i+2]])
                    single_text_label.sort(key=firstElement)
                true_text_labels.append(single_text_label)

        pred_text_labels = []
        with open(output_file+".tgt") as f:
            for line in f:
                raw_line = line.strip().split()
                assert len(raw_line) %3 == 0
                n_triple = len(raw_line) / 3
                single_text_label = []
                for i in range(n_triple):
                    single_text_label.append([int(raw_line[3*i][1:]), int(raw_line[3*i+1][1:]), raw_line[3*i+2]])
                    single_text_label.sort(key=firstElement)
                pred_text_labels.append(single_text_label)

        # Match predictions
        output_txt = []
        return_result = []
        for text, true_label, pred_label in zip(pred_texts, true_text_labels, pred_text_labels):
            text_tokens = text.split()
            single_text = []
            single_text_span = []
            char_count = 0
            for token in text_tokens:
                single_text.append([token, "O","O"])
                single_text_span.append([char_count, len(token)])
                char_count += len(token) + 1
            
            # Match true label
            # O(n) algorithm label each non-"O" token with tags.
            # It's complicated because the spans in the labels do not exact match with the position of the tokens.
            label_index = 0
            for i, token in enumerate(single_text):
                if label_index >= len(true_label):
                    break
                if single_text_span[i][0] >= true_label[label_index][0]:
                    shift = 0
                    while i+shift < len(single_text_span) and true_label[label_index][0]+true_label[label_index][1] > single_text_span[i+shift][0]:
                        single_text[i+shift][1] = true_label[label_index][2]
                        shift += 1
                    label_index += 1
            
            # Match pred label
            # O(n) algorithm label each non-"O" token with tags.
            # It's complicated because the spans in the labels do not exact match with the position of the tokens.
            label_index = 0
            for i, token in enumerate(single_text):
                if label_index >= len(pred_label):
                    break
                if single_text_span[i][0] >= pred_label[label_index][0]:
                    shift = 0
                    while i+shift < len(single_text_span) and pred_label[label_index][0]+pred_label[label_index][1] > single_text_span[i+shift][0]:
                        single_text[i+shift][2] = pred_label[label_index][2]
                        shift += 1
                    label_index += 1
            output_txt.append(single_text)
            
            for i,(single_piece, span) in enumerate(zip(single_text, single_text_span)):
                return_result.append((span[0], span[1], single_piece[0], single_piece[2]))
    
        with open(output_file+".txt","w") as f:
            f.write("WORD TRUE_LABEL PRED_LABEL\n")
            f.write("\n")
            for single_text in output_txt:
                for single_token in single_text:
                    f.write(" ".join(single_token)+"\n")
                f.write("\n")

        return return_result

    def evaluate(self, predictions, groundTruths):  # <--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
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
        pred = [element[2] for element in predictions]
        truth = [element[2] for element in groundTruths]
        precision = precision_score(truth, pred,average='weighted')
        recall = recall_score(truth, pred,average='weighted')
        f1 = f1_score(truth, pred,average='weighted')
        return (precision, recall, f1)

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

if __name__ == '__main__':

    file_dict = {
        "train": {"file 1":"examples/training.tsv"},
        "dev" : {"file 2":"examples/development.tsv"},
        "test" : {"file 3":"examples/evaluation.tsv"},
    }
    dataset_name = 'CONLL2003'
    # instatiate the class
    myModel = byteNER() 
    # read in a dataset for training
    data = myModel.read_dataset(file_dict, dataset_name)

    myModel.train(data,nb_epochs=1)  # trains the model and stores model state in object properties or similar
    
    predictions = myModel.predict(data)  # generate predictions! output format will be same for everyone
    test_labels = myModel.convert_ground_truth(data)  #<-- need ground truth labels need to be in same format as predictions!

    P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1
    print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
