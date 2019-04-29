# -*- coding: utf8 -*-
"""
Utility functions for data pre- and post-processing, creating vocabulary dictionaries, etc
"""

from subprocess import Popen, PIPE
from collections import OrderedDict

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
import cPickle as pickle
import re
import os
import codecs
import random
import numpy as np
import threading

import sys
reload(sys)
sys.setdefaultencoding('utf8')

models_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")
eval_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "evaluation")
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

def get_crf_proba(data_file, model_params):
    """
    Get intermediate outputs for entire dataset
    :param data_file:
    :param model_params:
    :return:
    """
    # open file
    m = model_params['model']
    ner_model = model_params['ner_model']
    max_chars_in_sample = str(model_params['max_chars_in_sample'])

    combined_ext = generate_combined_feat_ext(model_params)
    if len(combined_ext) > 0:
        X_file = data_file + combined_ext
    else:
        X_file = ''

    predictions = []
    probabilities = []
    vectors = []
    nb_workers = model_params['nb_workers']

    # generate word data (second input to model)
    X_word_file = ''
    if model_params['use_word_embeddings']:
        X_word_file = data_file + '.word-coded.' + str(max_chars_in_sample) + '.pkl'
    elif model_params['use_bpe_embeddings']:
        X_word_file = data_file + '.bpe-embed-coded-' + os.path.basename(model_params['bpe_codes_file']) + '.' + str(
            max_chars_in_sample) + '.pkl'

    data_gen_obj = GenBatchFromPickle(nb_workers)
    data_gen = data_gen_obj.thread_gen_batch_data_from_pickle(X_file, X_word_file)
    for batched_idx, batched_sample in enumerate(data_gen):
        preds = m.predict_on_batch(batched_sample)
        predictions.extend(preds)

        if model_params['get_probs']:
            crf_inputs = ner_model.get_intermediate_output_layer(-2, batched_sample, test_mode=True)
            probs = K.eval(ner_model.crf_predict_proba(preds, crf_inputs))
            probs = probs.reshape(probs.shape[0],)
            pred_lengths = np.array([len(p) for p in preds])
            norm_probs = probs / pred_lengths  # normalized log likelihood
            probabilities.extend(norm_probs)

        if model_params['get_vectors']:
            vector = ner_model.get_sample_vector(batched_sample)
            vectors.extend(vector)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    vectors = np.array(vectors)
    return predictions, probabilities, vectors


def load_data_stride_x_chars_enc_dec(data_file, parameters, stride=1):
    """
    Load data from pickled file with <[bytes]>\t<[(offset, length, entity_type)...]> data and annotations.
    Overlap x bytes from sample i-1 to sample i.
    :param data_file:
    :param parameters:
    :param stride
    :return:
    """
    X = []  # full data in batches
    y = []  # full annotations in batches
    sample = []
    progress = 0

    max_chars_in_sample = parameters['max_chars_in_sample']
    char_to_num = parameters['char_to_num']
    enc_dec_tag_to_num = parameters['enc_dec_tag_to_num']
    batch_size = parameters['batch_size']
    file_ext = parameters['file_ext']

    pickled_X_file = data_file + '.X.bytes.' + str(max_chars_in_sample) + file_ext
    pickled_y_file = data_file + '.y.' + str(max_chars_in_sample) + file_ext
    pickled_BPE_file = data_file + '.bpe.' + str(max_chars_in_sample) + file_ext
    # save complete words for each sample for BPE
    # we can cut off extraneous bytes for each sample after we've processed the BPE codes

    data = gen_paragraphs_from_pickle(data_file)

    if parameters['repickle_data'] or not (os.path.isfile(pickled_X_file) and os.path.isfile(pickled_y_file) and os.path.isfile(pickled_BPE_file)):

        with open(pickled_X_file, 'wb') as pX, open(pickled_y_file, 'wb') as py, open(pickled_BPE_file, 'wb') as pBPE:

            for lines_idx, lines in data:
                byte_list = lines[0]  # for sample to use
                annotation_list = lines[1]  # should be sorted from greatest to least based on offset

                for sentence_idx, sentence in enumerate(byte_list):
                    preprocessed_sentence_list = []  # for X to use
                    new_sentence = []
                    for b in sentence:
                        if b == ' ' or b == '\n':
                            preprocessed_sentence_list.append('<SPACE>')
                            new_sentence.append(' ')
                        else:
                            preprocessed_sentence_list.append(b)
                            new_sentence.append(b)
                    sentence = new_sentence

                    curr_char_idx = 0
                    while True:
                        # first sample has max_chars_in_sample bytes
                        if curr_char_idx + max_chars_in_sample <= len(sentence):
                            curr_X = preprocessed_sentence_list[curr_char_idx:curr_char_idx+max_chars_in_sample]
                            curr_sample = sentence[curr_char_idx:curr_char_idx+max_chars_in_sample]
                        else:
                            curr_X = preprocessed_sentence_list[curr_char_idx:]
                            curr_sample = sentence[curr_char_idx:]
                        progress += 1

                        curr_X = [char_to_num[c] for c in curr_X]
                        pad = max_chars_in_sample - len(curr_X)
                        curr_X += [0] * pad
                        X.append(curr_X)
                        sample.append(''.join(curr_sample))
                        # add all annotations that end within [curr_char_idx, cur_char_idx + max_chars_in_sample)
                        curr_y = []
                        sentence_annotations = annotation_list[sentence_idx]
                        for offset, entity_length, entity in sentence_annotations:  # either 10 or "S10"
                            if isinstance(offset, str):
                                # "S89" -> 89
                                offset = int(offset[1:])
                            if isinstance(entity_length, str):
                                # "L6" -> 6
                                entity_length = int(entity_length[1:])
                            if offset < curr_char_idx:  # annotated entity ends before current sample range
                                pass
                            elif offset >= curr_char_idx + max_chars_in_sample:  # annotated entity begins after current sample range
                                pass  # annotated entities are sorted in reverse by offset
                            else:  # annotated entity in current sample range
                                if offset + entity_length - 1 < curr_char_idx + max_chars_in_sample:
                                    curr_y.append(
                                        (offset - curr_char_idx, entity_length, entity))  # offset relative to this sample
                                else:
                                    pass
                        y.append(curr_y)

                        # save a batch of data
                        if len(X) == batch_size:
                            pickle.dump(np.array(X), pX, protocol=pickle.HIGHEST_PROTOCOL)
                            pickle.dump(y, py, protocol=pickle.HIGHEST_PROTOCOL)
                            pickle.dump(sample, pBPE, protocol=pickle.HIGHEST_PROTOCOL)
                            X = []
                            y = []
                            sample = []

                        if curr_char_idx + max_chars_in_sample >= len(sentence):
                            break

                        # for training data, we might want a smaller stride for more samples
                        # to evaluate dev and train data, we want half of a sample to overlap with previous sample
                        curr_char_idx += stride

            # put the rest of the residual data ( < batch_size samples ) into files
            if len(X) > 0:
                pickle.dump(np.array(X), pX, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(y, py, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(sample, pBPE, protocol=pickle.HIGHEST_PROTOCOL)

    if ('tag' not in parameters) or ('tag' in parameters and not parameters['tag']):
        # convert y to corresponding nums
        old_ys = gen_sentences_from_pickle(pickled_y_file)
        pickled_new_y_file = data_file + '.new.y.' + str(max_chars_in_sample) + '.sle' + file_ext
        all_new_ys = []
        with open(pickled_new_y_file, 'wb') as new_y_file:
            for batch_of_old_ys in old_ys:
                new_ys = []
                new_ys_tags = []
                for sample in batch_of_old_ys:
                    new_ys_sample = []
                    new_ys_sample_tags = []
                    for offset, entity_length, entity_type in sample:
                        if entity_length <= max_chars_in_sample:
                            offset_id = enc_dec_tag_to_num['S' + str(offset)]
                            entity_length_id = enc_dec_tag_to_num['L' + str(entity_length)]
                            entity_type_id = enc_dec_tag_to_num[entity_type]
                            new_ys_sample.append(offset_id)
                            new_ys_sample.append(entity_length_id)
                            new_ys_sample.append(entity_type_id)
                            new_ys_sample_tags.append('S' + str(offset))
                            new_ys_sample_tags.append('L' + str(entity_length))
                            new_ys_sample_tags.append(entity_type)
                        else:
                            print 'Entity length: ', entity_length, ' is longer than max chars in sample: ', max_chars_in_sample,'. Consider making max chars in sample larger.'
                            pass
                    new_ys_sample.append(enc_dec_tag_to_num['<STOP>'])
                    new_ys_tags.append(new_ys_sample_tags)
                    new_ys.append(new_ys_sample)
                pickle.dump(new_ys, new_y_file, protocol=pickle.HIGHEST_PROTOCOL)
                all_new_ys.append(new_ys_tags)  # save in sparse format (without padding)

        # put y file in IOB format too
        data = gen_sentences_from_pickle(pickled_X_file)
        pickled_new_y_iob_file = data_file + '.new.y.' + str(max_chars_in_sample) + '.iob' + file_ext
        with open(pickled_new_y_iob_file, 'wb') as new_y_iob_file:
            for batched_idx, batched_data in enumerate(data):
                curr_iob_output = convert_enc_dec_output_to_IOB(batched_data, all_new_ys[batched_idx], parameters)
                pickle.dump(curr_iob_output, new_y_iob_file, protocol=pickle.HIGHEST_PROTOCOL)

        # use iobes format for encoder-decoder
        pickled_new_y_file = data_file + '.new.y.' + str(max_chars_in_sample) + '.iobes' + file_ext
        # replace pickled_new_y_file with the following
        with open(pickled_new_y_file, 'wb') as new_y_file:
            data = gen_sentences_from_pickle(pickled_new_y_iob_file)
            for batched_data in data:
                new_batched_data = []
                for sample in batched_data:
                    new_sample = iob_iobes(sample)
                    tag_id_sample = []
                    for tag in new_sample:
                        tag_id_sample.append(parameters['tag_to_num'][tag])
                    new_batched_data.append(tag_id_sample)
                pickle.dump(new_batched_data, new_y_file)


def byte_dropout(data_file, parameters):
    """
    Randomly replace a fraction of byte inputs in each sample with special <DROP> symbol. (e.g., 0.3)
    :param data_file:
    :param parameters:
    :return:
    """
    random.seed(123)
    max_chars_in_sample = parameters['max_chars_in_sample']
    file_ext = parameters['file_ext']
    pickled_data_file = data_file + '.X.bytes.' + str(max_chars_in_sample) + file_ext
    drop_fraction = parameters['byte_drop_fraction']
    char_to_num = parameters['char_to_num']
    output_file = data_file + '.bytedrop.X.bytes.' + str(max_chars_in_sample) + file_ext

    if parameters['repickle_data'] or not os.path.isfile(output_file):
        with open(output_file, 'wb') as o:
            batched_data = gen_sentences_from_pickle(pickled_data_file)
            for sentences in batched_data:
                for sentence in sentences:
                    sample = random.sample(range(len(sentence)), len(sentence))
                    threshold = int(drop_fraction * len(sentence))
                    for i in range(len(sentence)):
                        if sample[i] < threshold and sentence[i]:  # drop
                            sentence[i] = char_to_num['<DROP>']
                        else:
                            pass
                pickle.dump(sentences, o)  # dump new samples with dropped data


def gen_bpe_code_file(data_file, num_operations, codes_file):
    """
    Run BPE on each generated batch of data.
    BPE algorithm needs to take all data at once.
    Save bpe data in another pickled file.
    :param parameters:
    :return:
    """
    data_gen = gen_sentences_from_pickle(data_file)
    train_bpe_array = []
    for sentence_batch in data_gen:
        train_bpe_array.extend(sentence_batch)
    train_input = '\n'.join(train_bpe_array)

    # run algorithm to learn byte pair encodings
    devnull = open(os.devnull, 'wb')
    p2 = Popen([os.path.abspath(os.path.dirname(__file__)) + "/subword_nmt/learn_bpe.py", "-s", str(num_operations)], stdin=PIPE, stdout=PIPE, stderr=devnull)
    stdout, stderr = p2.communicate(input=train_input)

    with open(codes_file, 'wb') as c:
        c.write(stdout)


def generate_combined_feat_ext(parameters):
    """
    Generate filename with desired combined feature inputs to model
    :param parameters:
    :return:
    """
    combined_data_ext = ''
    max_chars_in_sample = parameters['max_chars_in_sample']
    file_ext = parameters['file_ext']
    bpe_codes_file = parameters['bpe_codes_file']
    bpe_codes_file = os.path.basename(bpe_codes_file)

    if parameters['use_bytes']:
        if parameters['use_bpe']:
            if parameters['use_tokenization']:
                combined_data_ext = '.combined.X.bytes.bpe-' + bpe_codes_file + '.tok.' + str(max_chars_in_sample) + file_ext
            else:
                combined_data_ext = '.combined.X.bytes.bpe-' + bpe_codes_file + '.' + str(max_chars_in_sample) + file_ext
        else:
            if parameters['use_tokenization']:
                combined_data_ext = '.combined.X.bytes.tok.' + str(max_chars_in_sample) + file_ext
            else:
                combined_data_ext = '.combined.X.bytes.' + str(max_chars_in_sample) + file_ext
    elif parameters['use_bpe']:
        if parameters['use_tokenization']:
            combined_data_ext = '.combined.X.bpe-' + bpe_codes_file + '.tok.' + str(max_chars_in_sample) + file_ext
        else:
            combined_data_ext = '.combined.X.bpe-' + bpe_codes_file + '.' + str(max_chars_in_sample) + file_ext
    elif parameters['use_tokenization']:
        combined_data_ext = '.combined.X.tok.' + str(max_chars_in_sample) + file_ext
    else:
        if not parameters['use_word_embeddings'] and not parameters['use_bpe_embeddings']:
            print 'There are no input features!'
            exit()
    return combined_data_ext


def nertok_to_byte_iob(nertok_lines, sentence_batch, vocab_dict):
    """
    Convert from NERSuite tokenizer format to byte IOBES format.
    Extra spaces that occur before the last non-space character are accounted for by NERSuite tokenizer.
    Add in additional spaces that occur after the last non-space character.
    Tokenizer format:
        ...	...	...
        8	10	of
        11	14	ZAP
        14	15	-
    :param nertok_lines:
    :return:
    """
    data = []
    iob_data = []
    prev_end = 0
    sentence_batch_idx = 0

    for line_idx, line in enumerate(nertok_lines):
        line = line.strip()
        if len(line) > 0:
            split = line.split('\t')
            start = int(split[0])
            end = int(split[1])
            curr_start = start

            # whitespace in tokenization
            while prev_end < start:
                iob_data.append(vocab_dict['O-TOK'])
                prev_end += 1
            prev_end = end

            while curr_start < end:
                # single
                if end - start == 1:
                    iob_data.append(vocab_dict['S-TOK'])
                else:
                    if curr_start == start:
                        iob_data.append(vocab_dict['B-TOK'])
                    elif curr_start == end - 1:
                        iob_data.append(vocab_dict['E-TOK'])
                    else:
                        iob_data.append(vocab_dict['I-TOK'])
                curr_start += 1
        else:
            num_spaces = 0
            if len(iob_data) < len(sentence_batch[sentence_batch_idx]):
                # append extra spaces found at the end of sentence
                num_spaces = len(sentence_batch[sentence_batch_idx]) - len(sentence_batch[sentence_batch_idx].rstrip())
                iob_data += [vocab_dict['O-TOK']] * num_spaces
            if len(iob_data) != len(sentence_batch[sentence_batch_idx]):
                print iob_data
                print sentence_batch[sentence_batch_idx]
                print len(iob_data)
                print len(sentence_batch[sentence_batch_idx])
            assert(len(iob_data) == len(sentence_batch[sentence_batch_idx]))
            data.append(iob_data)
            iob_data = []
            sentence_batch_idx += 1
            prev_end = end + num_spaces + 1  # for next sample

    if len(iob_data) > 0:
        if len(iob_data) < len(sentence_batch[sentence_batch_idx]):
            # append extra spaces found at the end of sentence
            num_spaces = len(sentence_batch[sentence_batch_idx]) - len(iob_data)
            iob_data += [vocab_dict['O-TOK']] * num_spaces
        assert (len(iob_data) == len(sentence_batch[sentence_batch_idx]))
        data.append(iob_data)

    return data


def load_tok_data(data_file, vocab_dict, output_file, parameters):
    """
    Load tokenization data from data_file
    Save formatted data into file for generator.
    :param data_file:
    :param vocab_dict:
    :param output_file:
    :param parameters:
    :return:
    """
    max_chars_in_sample = parameters['max_chars_in_sample']
    data_gen = gen_sentences_from_pickle(data_file)
    if parameters['repickle_data'] or not os.path.isfile(output_file):
        with open(output_file, 'wb') as o:
            for sentence_batch in data_gen:
                sentence_batch_replaced = []
                for s in sentence_batch:
                    # unicode to ascii
                    decoded = s.decode('utf8', 'replace')
                    replaced = decoded.replace(u'\ufffd', '*')
                    replaced = bytes.encode(str(replaced))
                    sentence_batch_replaced.append(replaced)
                sentence_batch = sentence_batch_replaced

                sentence_batch_str = '\n'.join(sentence_batch)
                p2 = Popen(['nersuite_tokenizer'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                stdout, stderr = p2.communicate(input=sentence_batch_str)
                if stderr:
                    print stderr
                stdout = stdout.strip()
                stdout_lines = stdout.split('\n')

                nertok_data = nertok_to_byte_iob(stdout_lines, sentence_batch, vocab_dict)

                if len(sentence_batch) != len(nertok_data):
                    print 'len sentence batch', len(sentence_batch)
                    print 'len nertok_data', len(nertok_data)
                    print 'sentence_batch', sentence_batch
                    print 'nertok_data', nertok_data
                assert (len(sentence_batch) == len(nertok_data))

                # find corresponding repr in vocab_dict or create new entry
                batch_values = []
                for sample_data in nertok_data:
                    # pad if necessary
                    if len(sample_data) < max_chars_in_sample:
                        num_to_pad = max_chars_in_sample - len(sample_data)
                        sample_data += [0] * num_to_pad
                    elif len(sample_data) > max_chars_in_sample:
                        sample_data = sample_data[:max_chars_in_sample]
                    if len(sample_data) != max_chars_in_sample:
                        print 'len sample data', len(sample_data)
                    assert(len(sample_data) == max_chars_in_sample)
                    batch_values.append(sample_data)

                pickle.dump(batch_values, o)


def load_bpe_data(data_file, vocab_dict, codes_file, output_file, parameters):
    """
    Load BPE data from data_file
    Save formatted data into file for generator.
    :param data_file:
    :param parameters:
    :return:
    """
    max_chars_in_sample = parameters['max_chars_in_sample']
    data_gen = gen_sentences_from_pickle(data_file)
    devnull = open(os.devnull, 'wb')
    if parameters['repickle_data'] or not os.path.isfile(output_file):
        with open(output_file, 'wb') as o:
            for sentence_batch in data_gen:
                sentence_batch_replaced = []
                for s in sentence_batch:
                    replaced = bytes.encode(str(s.decode('utf8', 'replace')))
                    sentence_batch_replaced.append(replaced)
                sentence_batch = sentence_batch_replaced

                batch_values = []
                sentence_batch_str = '\n'.join(sentence_batch).strip()
                sentence_batch = sentence_batch_str.split('\n')
                # make sure sentence_batch does not have empty strings
                sentence_batch = [x for x in sentence_batch if x]
                p2 = Popen([os.path.abspath(os.path.dirname(__file__)) + "/subword-nmt/apply_bpe.py", "-c", codes_file], stdin=PIPE, stdout=PIPE, stderr=devnull)
                stdout, stderr = p2.communicate(input=sentence_batch_str)
                stdout = bytes.encode(stdout.strip())
                stdout_sentences = stdout.split('\n')
                stdout_sentences = [x for x in stdout_sentences if x]

                if len(sentence_batch) != len(stdout_sentences):
                    print 'len sentence batch', len(sentence_batch)
                    print 'len stdout_sentences', len(stdout_sentences)
                    print 'sentence_batch', sentence_batch
                    print 'stdout_sentences', stdout_sentences
                assert (len(sentence_batch) == len(stdout_sentences))

                # find corresponding repr in vocab_dict or create new entry
                for idx, (sentence, stdout_sentence) in enumerate(zip(sentence_batch, stdout_sentences)):
                    curr_batch = []
                    stdout_sentence_split = stdout_sentence.split()
                    sentence_idx = 0
                    sentence = bytes.encode(sentence)

                    for token_idx, old_token in enumerate(stdout_sentence_split):

                        # append BPE <SPACE> representations
                        while sentence[sentence_idx].isspace():
                            curr_batch.append(vocab_dict['<SPACE>'])
                            sentence_idx += 1

                        old_token = bytes.encode(old_token)
                        bare_token = old_token.replace('@@', '')

                        if old_token.endswith('@@'):  # middle of a word
                            token = old_token.replace('@@', '')
                            if token in vocab_dict:
                                for _ in range(len(bare_token)):
                                    curr_batch.append(vocab_dict[token])
                            else:  # ex: old_token = °@@
                                for byt in token:
                                    assert(byt in vocab_dict)  # all bytes should be in vocab_dict
                                    curr_batch.append(vocab_dict[byt])
                        else:
                            token = old_token + '</w>'  # try looking for more specific match first (with EOW symbol)
                            if token in vocab_dict:
                                for _ in range(len(bare_token)):
                                    curr_batch.append(vocab_dict[token])
                            else:
                                token = old_token  # now look for more general match (with no EOW symbol)
                                if token in vocab_dict:
                                    for _ in range(len(bare_token)):
                                        curr_batch.append(vocab_dict[token])
                                else:  # ex: old_token = °
                                    for byt in token:
                                        assert (byt in vocab_dict)  # all bytes should be in vocab_dict
                                        curr_batch.append(vocab_dict[byt])
                        sentence_idx += len(bare_token)

                    while sentence_idx < len(sentence):  # whitespace at end of sentence
                        byt = sentence[sentence_idx]
                        if byt.isspace():
                            embed = vocab_dict['<SPACE>']
                        else:
                            assert (byt in vocab_dict)
                            embed = vocab_dict[byt]
                        curr_batch.append(embed)
                        sentence_idx += 1

                    # pad if necessary
                    if len(curr_batch) < max_chars_in_sample:
                        num_to_pad = max_chars_in_sample - len(curr_batch)
                        curr_batch += [0] * num_to_pad
                    # get rid of extraneous bytes if necessary
                    elif len(curr_batch) > max_chars_in_sample:
                        curr_batch = curr_batch[:max_chars_in_sample]
                    assert(len(curr_batch) == max_chars_in_sample)
                    batch_values.append(curr_batch)

                pickle.dump(batch_values, o)


def combine_data(data_file, combined_data_ext, parameters):
    """
    Combine byte, BPE, tok data.
    Update train_data_file, dev_data_file, and test_data_file in parameters to point to updated files.
    :param parameters:
    :return:
    """
    if combined_data_ext != '':
        max_chars_in_sample = parameters['max_chars_in_sample']
        file_ext = parameters['file_ext']

        gen = GenBatchFromPickle()

        # byte data location
        if parameters['use_bytes']:
            X_byte_file = data_file + '.X.bytes.' + str(max_chars_in_sample) + file_ext
            byte_gen = gen.thread_gen_batch_data_from_pickle(X_byte_file)

        # bpe data location
        if parameters['use_bpe']:
            bpe_file = data_file.replace('.bytedrop', '') + '.bpe-coded-' + os.path.basename(parameters['bpe_codes_file']) + '.' + str(max_chars_in_sample) + file_ext
            bpe_gen = gen.thread_gen_batch_data_from_pickle(bpe_file)

        if parameters['use_tokenization']:
            tok_file = data_file.replace('.bytedrop', '') + '.tok.' + str(max_chars_in_sample) + file_ext
            tok_gen = gen.thread_gen_batch_data_from_pickle(tok_file)

        combined_data_file = data_file + combined_data_ext

        # combine data
        if parameters['repickle_data'] or not os.path.isfile(combined_data_file):
            with open(combined_data_file, 'wb') as combined_data:
                try:
                    while True:
                        byte_batch = None
                        bpe_batch = None
                        tok_batch = None
                        num_samples = 0

                        if parameters['use_bytes']:
                            byte_batch = byte_gen.next()[0]
                            num_samples = len(byte_batch)
                        if parameters['use_bpe']:
                            bpe_batch = bpe_gen.next()[0]
                            num_samples = len(bpe_batch)
                        if parameters['use_tokenization']:
                            tok_batch = tok_gen.next()[0]
                            num_samples = len(tok_batch)

                        # Check that batches of different inputs line up
                        if byte_batch is not None:
                            if bpe_batch is not None:
                                if tok_batch is not None:
                                    assert (len(byte_batch) == len(bpe_batch) == len(tok_batch))
                                    batches = [byte_batch, bpe_batch, tok_batch]
                                else:
                                    assert(len(byte_batch) == len(bpe_batch))
                                    batches = [byte_batch, bpe_batch]
                            else:
                                if tok_batch is not None:
                                    assert(len(byte_batch) == len(tok_batch))
                                    batches = [byte_batch, tok_batch]
                                else:
                                    batches = [byte_batch]
                        elif bpe_batch is not None:
                            if tok_batch is not None:
                                assert (len(bpe_batch) == len(tok_batch))
                                batches = [bpe_batch, tok_batch]
                            else:
                                batches = [bpe_batch]
                        else:
                            if tok_batch is not None:
                                batches = [tok_batch]

                        new_batch = []
                        for i in range(num_samples):
                            sentence_batch = []
                            for input_batch in batches:
                                assert (len(input_batch[i]) == len(batches[0][i]))
                                sentence_batch.append(input_batch[i])

                            new_sentence = []
                            for byt_idx in range(len(sentence_batch[0])):
                                byte_input = []
                                for sentence in sentence_batch:
                                    byte_input.append(sentence[byt_idx])
                                new_sentence.append(tuple(byte_input))
                            new_batch.append(new_sentence)

                        new_batch = np.array(new_batch)
                        pickle.dump(new_batch, combined_data)

                except StopIteration:
                        pass


def load_word_embeddings_data(data_file, vocab_dict, output_file, parameters):
    """
    Load word embeddings data
    Save formatted data into file for generator
    """
    max_chars_in_sample = parameters['max_chars_in_sample']
    if parameters['repickle_data'] or not os.path.isfile(output_file):
        data_gen = gen_sentences_from_pickle(data_file)
        with open(output_file, 'wb') as o:
            for sentence_batch in data_gen:
                sentence_batch_replaced = []
                for s in sentence_batch:
                    replaced = bytes.encode(str(s.decode('utf8', 'replace')))
                    sentence_batch_replaced.append(replaced)
                sentence_batch = sentence_batch_replaced
                batch_values = []

                # find corresponding repr in vocab_dict or create new entry
                for idx, sentence in enumerate(sentence_batch):
                    curr_batch = []
                    sentence_split = sentence.split()
                    sentence_idx = 0

                    for word_idx, word in enumerate(sentence_split):

                        # append BPE <SPACE> representations
                        while sentence[sentence_idx].isspace():
                            curr_batch.append([vocab_dict['<SPACE>'][0]])
                            sentence_idx += 1

                        byte_word = bytes.encode(word)

                        if byte_word in vocab_dict:
                            vocab_num = vocab_dict[byte_word][0]
                        else:  # assign UNK embedding
                            # print byte_word, 'not found'
                            vocab_num = vocab_dict['<UNKNOWN>'][0]

                        for _ in range(len(byte_word)):
                            curr_batch.append([vocab_num])
                        sentence_idx += len(byte_word)

                    while sentence_idx < len(sentence):  # whitespace at end of sentence
                        byt = sentence[sentence_idx]
                        if byt.isspace():
                            embed = [vocab_dict['<SPACE>'][0]]
                        elif byt in vocab_dict:
                            embed = [vocab_dict[byt][0]]
                        else:
                            embed = [vocab_dict['<UNKNOWN>'][0]]
                        curr_batch.append(embed)
                        sentence_idx += 1

                    # pad if necessary
                    if len(curr_batch) < max_chars_in_sample:
                        num_to_pad = max_chars_in_sample - len(curr_batch)
                        curr_batch += [[vocab_dict['<PADDING>'][0]]] * num_to_pad
                    # get rid of extraneous bytes if necessary (possible if we're using the encoder-decoder model
                    elif len(curr_batch) > max_chars_in_sample:
                        curr_batch = curr_batch[:max_chars_in_sample]
                    assert (len(curr_batch) == max_chars_in_sample)
                    batch_values.append(curr_batch)

                batch_values = np.array(batch_values)
                pickle.dump(batch_values, o)


def load_bpe_embeddings_data(data_file, vocab_dict, codes_file, output_file, parameters):
    """
    (For unknown bpe tokens, we will count how many there are and use UNK)

    Load bpe embeddings data
    Save formatted data into file for generator
    """
    max_chars_in_sample = parameters['max_chars_in_sample']
    num_unk = 0
    if parameters['repickle_data'] or not os.path.isfile(output_file):
        data_gen = gen_sentences_from_pickle(data_file)
        with open(output_file, 'wb') as o:
            for sentence_batch in data_gen:
                sentence_batch_replaced = []
                for s in sentence_batch:
                    replaced = bytes.encode(str(s.decode('utf8', 'replace')))
                    sentence_batch_replaced.append(replaced)
                sentence_batch = sentence_batch_replaced

                batch_values = []
                sentence_batch_str = '\n'.join(sentence_batch)  # if extra whitespace, bpe code will disregard
                sentence_batch = sentence_batch_str.split('\n')
                sentence_batch = [x for x in sentence_batch if x]
                p2 = Popen([os.path.abspath(os.path.dirname(__file__)) + "/subword_nmt/apply_bpe.py", "-c", codes_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                stdout, stderr = p2.communicate(input=sentence_batch_str)
                if stderr:
                    print stderr
                stdout = bytes.encode(stdout.strip())
                stdout_sentences = stdout.split('\n')
                stdout_sentences = [x for x in stdout_sentences if x]

                if len(sentence_batch) != len(stdout_sentences):
                    print 'len sentence batch', len(sentence_batch)
                    print 'len stdout_sentences', len(stdout_sentences)
                    print 'sentence_batch', sentence_batch
                    print 'stdout_sentences', stdout_sentences
                assert (len(sentence_batch) == len(stdout_sentences))

                # find corresponding repr in vocab_dict or create new entry
                for idx, (sentence, stdout_sentence) in enumerate(zip(sentence_batch, stdout_sentences)):
                    curr_batch = []
                    stdout_sentence_split = stdout_sentence.split()
                    sentence_idx = 0
                    sentence = bytes.encode(sentence)

                    for token_idx, old_token in enumerate(stdout_sentence_split):

                        # append BPE <SPACE> representations
                        while sentence[sentence_idx].isspace():
                            curr_batch.append([vocab_dict['<SPACE>'][0]])
                            sentence_idx += 1

                        old_token = bytes.encode(old_token)
                        bare_token = old_token.replace('@@', '')

                        if old_token.endswith('@@'):  # middle of a word
                            token = old_token.replace('@@', '')
                            if token in vocab_dict:
                                for _ in range(len(bare_token)):
                                    curr_batch.append([vocab_dict[token][0]])
                            else:  # ex: old_token = °@@ -> UNK
                                # print token, 'not found'
                                for _ in range(len(bare_token)):
                                    num_unk += 1
                                    curr_batch.append([vocab_dict['<UNKNOWN>'][0]])
                            sentence_idx += len(bare_token)
                        else:
                            token = old_token + '</w>'  # try looking for more specific match first (with EOW symbol)
                            if token in vocab_dict:
                                for _ in range(len(bare_token)):
                                    curr_batch.append([vocab_dict[token][0]])
                            else:
                                token = old_token  # now look for more general match (with no EOW symbol)
                                if token in vocab_dict:
                                    for _ in range(len(bare_token)):
                                        curr_batch.append([vocab_dict[token][0]])
                                else:  # ex: old_token = ° -> UNK
                                    # print token, 'not found'
                                    for _ in range(len(bare_token)):
                                        num_unk += 1
                                        curr_batch.append([vocab_dict['<UNKNOWN>'][0]])
                            sentence_idx += len(bare_token)

                    while sentence_idx < len(sentence):  # whitespace at end of sentence
                        byt = sentence[sentence_idx]
                        if byt.isspace():
                            embed = [vocab_dict['<SPACE>'][0]]
                        elif byt in vocab_dict:
                            embed = [vocab_dict[byt][0]]
                        else:
                            embed = [vocab_dict['<UNKNOWN>'][0]]
                        curr_batch.append(embed)
                        sentence_idx += 1

                    # pad if necessary
                    if len(curr_batch) < max_chars_in_sample:
                        num_to_pad = max_chars_in_sample - len(curr_batch)
                        curr_batch += [[vocab_dict['<PADDING>'][0]]] * num_to_pad
                    # get rid of extraneous bytes if necessary (possible if we're using the encoder-decoder model
                    elif len(curr_batch) > max_chars_in_sample:
                        curr_batch = curr_batch[:max_chars_in_sample]
                    assert (len(curr_batch) == max_chars_in_sample)
                    batch_values.append(curr_batch)

                batch_values = np.array(batch_values)
                pickle.dump(batch_values, o)

    print '# unknown chunks not in bpe embedding vocab:', num_unk


def add_bpe_to_vocab_dictionary(codes_file, char_to_num):
    """
    Add new BPE tokens to char_to_num vocabulary dictionary
    :param codes_file:
    :param char_to_num:
    :return:
    """
    with open(codes_file, 'rb') as c:
        line_idx = 0
        for line_idx, line in enumerate(c):
            if line_idx >= 2:
                line = bytes.encode(line)
                line = line.strip()
                line = line.replace(' ', '')
                char_to_num[line] = len(char_to_num)
        print 'Found %s BPE tokens' % (line_idx - 1)


def add_embeddings_to_vocab_dictionary(embeddings_file, vocab_dict, parameters):
    """
    Add new word embedding words to char_to_num vocabulary dictionary
    """
    with open(embeddings_file, 'rb') as f:
        line_idx = 0
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                if parameters['use_word_embeddings']:
                    dim = parameters['word_embeddings_dim']
                elif parameters['use_bpe_embeddings']:
                    dim = parameters['bpe_embeddings_dim']
                else:
                    print 'No specified embeddings to add to vocab dict'
                    exit()
                vocab_dict['<PADDING>'] = (line_idx, np.zeros(dim))
            else:
                values = line.strip().split()
                tok = values[0]
                embed = np.asarray(values[1:], dtype='float32')
                if tok not in vocab_dict:
                    vocab_dict[tok] = (line_idx, embed)
        print 'Found %s embeddings' % (line_idx + 1)


def gen_paragraphs_from_pickle(data_file):
    """
    Create a generator to yield raw data from pickled file.
    :param pickle_file:
    :return:
    """
    with open(data_file, 'rb') as d:
        paragraphs = pickle.load(d)
        for line_idx, lines in enumerate(paragraphs):
            yield (line_idx, lines)


def gen_sentences_from_pickle(pickled_sentences_file):
    """
    Create a generator to yield batches of sentences.
    :param pickled_sentences_file:
    :return:
    """
    with open(pickled_sentences_file, 'rb') as p:
        while True:
            try:
                sentences = pickle.load(p)
                yield sentences
            except EOFError:
                return


class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class GenBatchForFit:
    """
    Generate X and y batches for fitting models.
    """

    def __init__(self, nb_workers=1):
        self.nb_workers = nb_workers

    @threadsafe_generator
    def thread_gen_batch_data_for_fit(self, pickled_X_file, pickled_y_file, params, pickled_X_word_file=''):
        """
        Generate batched data in a format compatible with Keras' model.fit_generator. For thread-based (non-multiprocessing) computations.
        :param pickled_file:
        :return:
        """
        if len(pickled_X_word_file) > 0:  # word embeddings present in input
            while True:
                batch_idx = 0
                if len(pickled_X_file) > 0:
                    X_data = open(pickled_X_file, 'rb')
                with open(pickled_y_file, 'rb') as y_data, open(pickled_X_word_file, 'rb') as X_word:
                    while True:
                        try:
                            if len(pickled_X_file) > 0:
                                batch_X_data = pickle.load(X_data)
                            else:
                                batch_X_data = []
                            batch_y_data = pickle.load(y_data)
                            batch_X_word = pickle.load(X_word)
                            batch_y_data, decoder_input = self.load_dense_y(batch_y_data, params)

                            if len(batch_X_data) > 0:
                                yield([batch_X_data, batch_X_word], batch_y_data)
                            else:
                                yield ([batch_X_word], batch_y_data)
                            batch_idx += 1
                        except EOFError:
                            break
                    continue
                X_data.close()
        else:  # no word embeddings in input
            while True:
                batch_idx = 0
                with open(pickled_X_file, 'rb') as X_data, open(pickled_y_file, 'rb') as y_data:
                    while True:
                        try:
                            batch_X_data = pickle.load(X_data)
                            batch_y_data = pickle.load(y_data)
                            batch_y_data, decoder_input = self.load_dense_y(batch_y_data, params)
                            yield ([batch_X_data], batch_y_data)
                            batch_idx += 1
                        except EOFError:
                            break
                    continue

    def load_dense_y(self, batch_y_data, params):
        """
        Load sparse y into dense format.
        :param batch_y_data:
        :param params:
        :return:
        """
        new_batch_y_data = []
        decoder_input = []
        tag_to_num = params['tag_to_num']

        for y in batch_y_data:
            y = to_categorical(y, num_classes=len(tag_to_num))
            new_batch_y_data.append(y)

        new_batch_y_data = np.array(new_batch_y_data)
        decoder_input = np.array(decoder_input)
        return new_batch_y_data, decoder_input


class GenBatchFromPickle:
    """
    Generate batches of X or y data from pickled files.
    """

    def __init__(self, nb_workers=1):
        self.nb_workers = nb_workers

    @threadsafe_generator
    def thread_gen_batch_data_from_pickle(self, pickled_file, pickled_word_file=''):
        """
        Generator from pickled file. For thread-based (non-multiprocessing) computations.
        https://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-pickle-file
        :param pickled_file:
        :return:
        """
        if len(pickled_word_file) > 0:  # word embeddings present in input
            f = None
            if len(pickled_file) > 0:
                f = open(pickled_file, 'rb')
            with open(pickled_word_file, 'rb') as word_f:
                batch_idx = 0
                while True:
                    try:
                        batch_word = pickle.load(word_f)
                        if len(pickled_file) > 0:
                            batch_data = pickle.load(f)
                        else:
                            batch_data = []
                        if len(batch_data) > 0:
                            yield [batch_data, batch_word]
                        else:
                            yield [batch_word]
                        batch_idx += 1
                    except EOFError:
                        break
            if f:
                f.close()
        else:  # no word embeddings in input
            with open(pickled_file, 'rb') as f:
                batch_idx = 0
                while True:
                    try:
                        batch_data = pickle.load(f)
                        yield [batch_data]
                        batch_idx += 1
                    except EOFError:
                        break


def calc_batch_steps_size(pickled_file, batch_size):
    """
    Calculate the total number of steps (batches of samples) to yield from generator before declaring one epoch of unique samples finished.
    :param pickled_file:
    :return:
    """
    gen_obj = GenBatchFromPickle()
    data = gen_obj.thread_gen_batch_data_from_pickle(pickled_file)
    count = 0
    d_feature = None
    for d_gen in data:
        for d_feature_idx, d_feature in enumerate(d_gen):
            if d_feature_idx == 0:  # don't need to recount different features of same samples
                count += 1
        residual = len(d_feature)
    # print 'Count:', count
    if count == 0:
        print 'No data found! (Try deleting generated pickled files)'
        exit()
    assert(count != 0)
    # print 'Residual samples:', residual
    # print '# SAMPLES:', ((count - 1) * batch_size) + residual
    return count


def build_char_to_num():
    """
    Map each byte to 256-vector space.
    :return:
    """
    char_to_num = OrderedDict()
    char_to_num['<PADDING>'] = 0
    for i in range(1, 256):
        if chr(i) == ' ':
            k = '<SPACE>'
        else:
            k = chr(i)
        char_to_num[k] = i

    return char_to_num


def build_tok_char_to_num():
    """
    Map tokenization features to ids
    :return:
    """
    char_to_num = OrderedDict()
    char_to_num['I-TOK'] = len(char_to_num)
    char_to_num['O-TOK'] = len(char_to_num)
    char_to_num['B-TOK'] = len(char_to_num)
    char_to_num['E-TOK'] = len(char_to_num)
    char_to_num['S-TOK'] = len(char_to_num)
    return char_to_num


def build_enc_dec_tag_to_num(params):
    """
    Map possible tags to nums
    :param params:
    :return:
    """
    tag_to_num = OrderedDict()
    idx = 0
    for i in range(params['max_chars_in_sample']):
        tag_to_num['S' + str(i)] = idx
        idx += 1
        tag_to_num['L' + str(i + 1)] = idx
        idx += 1
    for tag in params['tags']:
        tag_to_num[tag] = idx
        idx += 1
    tag_to_num['<STOP>'] = idx
    idx += 1
    tag_to_num['<GO>'] = idx  # for teacher forcing

    return tag_to_num


def build_tag_to_num(tags):
    """
    Map each IOBES tag to an integer.
    :return:
    """
    tag_to_num = {}
    idx = 0
    tag_to_num['O'] = idx
    idx += 1
    for tag in tags:
        tag_to_num['B-' + tag] = idx
        idx += 1
        tag_to_num['I-' + tag] = idx
        idx += 1
        tag_to_num['E-' + tag] = idx
        idx += 1
        tag_to_num['S-' + tag] = idx
        idx += 1
    return tag_to_num


def build_num_to_tags(tag_to_num):
    """
    Reverse tag_to_num dictionary.
    :param tag_to_num:
    :return:
    """
    num_to_tags = {}
    for tag in tag_to_num:
        num = tag_to_num[tag]
        num_to_tags[num] = tag
    return num_to_tags


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    Taken from https://github.com/glample/tagger.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    Taken from https://github.com/glample/tagger.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    Taken from https://github.com/glample/tagger.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    Taken from https://github.com/glample/tagger.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    Taken from https://github.com/glample/tagger.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


class MetricsCheckpoint(ModelCheckpoint):
    """
    CoNLL metrics at chunk level
    """
    def __init__(self, filepath, parameters, monitor='conll_chunk_metric', verbose=1, save_best_only=True, save_weights_only=False, mode='max'):
        ModelCheckpoint.__init__(self, filepath=filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only, mode=mode)
        self.parameters = parameters
        if 'input' in self.parameters:  # inference
            self.test_X_data, _, _= self.generate_evaluation_values(
                parameters['input'],
                parameters)
        else:  # training and evaluating
            self.dev_X_data, self.dev_r_tags, self.dev_y_reals = self.generate_evaluation_values(parameters['dev_data_file'], parameters)
            self.test_X_data, self.test_r_tags, self.test_y_reals = self.generate_evaluation_values(parameters['test_data_file'],
                                                                                        parameters)

    def on_epoch_end(self, epoch, logs=None):
        """
        Perform at the end of every epoch
        :param epoch:
        :param logs:
        :return:
        """
        # save model based on CoNLL chunk F1 score
        print 'Evaluating on dev data'
        conll_chunk_f1 = self.evaluate(self.parameters['dev_data_file'], self.dev_X_data, self.dev_r_tags, self.dev_y_reals, self.parameters)

        logs = {'conll_chunk_metric': conll_chunk_f1}

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)

                        # Print CoNLL chunk test F1 score
                        print 'Evaluating on test data'
                        self.evaluate(self.parameters['test_data_file'], self.test_X_data, self.test_r_tags, self.test_y_reals, self.parameters)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

    def generate_evaluation_values(self, data_file, parameters):
        """
        Generate values needed for evaluation (to save them in metrics object)
        :param data_file:
        :param parameters:
        :return:
        """
        tag_to_num = parameters['tag_to_num']
        num_to_tags = build_num_to_tags(tag_to_num)
        max_chars_in_sample = str(parameters['max_chars_in_sample'])
        file_ext = parameters['file_ext']

        # format character data
        print 'formatting X data for evaluation...'
        X_file = data_file + '.X.bytes.' + str(max_chars_in_sample) + '.pkl'
        gen_obj = GenBatchFromPickle()
        X_gen = gen_obj.thread_gen_batch_data_from_pickle(X_file, '')
        X_data = []
        for X_batch_sample_idx, X_batched_sample in enumerate(X_gen):
            for X_gen_feature_idx, X_gen_feature in enumerate(X_batched_sample):
                for X_sample_idx, X_sample in enumerate(X_gen_feature):
                    new_X_sample = []
                    for x in X_sample:
                        new_X_sample.append([str(x)])
                    X_data.append(new_X_sample)

        # format real tags for CoNLL script
        # this should always be in IOB format
        r_tags = []
        y_reals = []
        if 'input' not in parameters:
            print 'formatting real y tags for evaluation...'
            y_file = data_file + '.new.y.' + max_chars_in_sample + '.iobes' + file_ext
            y_gen = gen_sentences_from_pickle(y_file)
            for y_batched_sample in y_gen:
                for y_sample in y_batched_sample:
                    tag_ids = []
                    for num in y_sample:
                        tag_ids.append(num_to_tags[num])
                    iob_tag_ids = iobes_iob(tag_ids)
                    y_reals.append(y_sample)
                    r_tags.append(iob_tag_ids)

        return X_data, r_tags, y_reals

    def make_and_format_predictions(self, data_file, X_data, parameters):
        """
        Make and format predictions for tagger
        :return:
        """
        tag_to_num = parameters['tag_to_num']
        predictions = []
        prediction_probabilities = []
        num_to_tags = build_num_to_tags(tag_to_num)

        # format y_preds
        print 'Making predictions...'
        train_predictions, probabilities, vectors = get_crf_proba(data_file, parameters)

        print 'Formatting predictions...'
        p_tags = []
        y_preds = []
        p_preds = []  # for each byte: {tag: prob, tag2: prob2, ...}
        for train_idx, train_sample in enumerate(train_predictions):
            tag_ids = train_sample.argmax(axis=1)
            y_preds.append(tag_ids)
            train_sample_tags = []
            train_sample_preds = []
            for tag_id in tag_ids:
                train_sample_tags.append(num_to_tags[tag_id])
            if parameters['tag_scheme'] == 'iobes':
                train_sample_tags = iobes_iob(train_sample_tags)
            p_tags.append(train_sample_tags)
            for sample in train_sample:  # train_sample is 2D, sample is 1D
                tag_to_pred = []
                for tag_idx in range(len(sample)):
                    tag_to_pred.append((num_to_tags[tag_idx], sample[tag_idx]))
                train_sample_preds.append(str(tag_to_pred))
            p_preds.append(train_sample_preds)

        del train_predictions

        # clean X_data, p_tags, y_preds, r_tags, y_reals so that there are no duplicate data
        # if entity starts in one sample and ends in another, overwrite earlier sample prediction
        print 'Combine overlapping data...'
        X_data, _, _, p_tags, y_preds, probabilities, vectors = self.combine_overlapping_data(X_data, None, None, p_tags,
                                                                                 y_preds, probabilities, vectors, parameters)

        # combine vector to be per sample
        combined_vectors = []
        for vector in vectors:  # (# samples, x)
            vector = np.sum(vector, axis=0)
            combined_vectors.append(vector)
        vectors = combined_vectors

        print 'Gathering and writing predictions...'
        for i, y_pred_sample in enumerate(y_preds):
            X_sample_nums = []
            for j, y_pred in enumerate(y_pred_sample):
                feat_id = str(X_data[i][j][0]).split(',')[0]
                if int(feat_id):  # not the byte for padding
                    X_sample_num = ','.join(X_data[i][j])  # byte features of this sample
                    predictions.append(' '.join([str(X_sample_num), p_tags[i][j]]))
                    # format prediction_probabilities
                    X_sample_nums.append(int(feat_id))
            predictions.append('')

            if parameters['get_probs'] or parameters['get_vectors']:
                X_sample_nums = ''.join([chr(x) for x in X_sample_nums])
                X_sample_nums = X_sample_nums.replace('\t', ' ')
                curr_line = [X_sample_nums]
                if parameters['get_probs']:
                    curr_line.append(str(probabilities[i]))
                if parameters['get_vectors']:
                    vector_strs = []
                    for vector in vectors:
                        vector_strs.append(' '.join([str(v) for v in vector]))
                    curr_line.append(vector_strs[i])
                prediction_probabilities.append('\t'.join(curr_line))

        del X_data
        del p_tags
        del y_preds

        # Write predictions to disk
        with codecs.open(parameters['output'], 'wb') as f:
            print 'writing...'
            f.write("\n".join(predictions))  # write feature ids (e.g., byte ids, bpe ids)

        # Write probabilities file
        if parameters['get_probs'] or parameters['get_vectors']:
            with codecs.open(parameters['output'] + '.probs_vectors', 'wb') as f:
                print 'writing probabilities...'
                f.write('\n'.join(prediction_probabilities) + '\n')

    def evaluate(self, data_file, X_data, r_tags, y_reals, parameters):
        """
        Evaluate current model using CoNLL script.
        Modified from https://github.com/glample/tagger.
        """
        tag_to_num = parameters['tag_to_num']
        n_tags = len(tag_to_num)
        predictions = []
        prediction_probabilities = []
        count = np.zeros((n_tags, n_tags), dtype=np.int32)
        num_to_tags = build_num_to_tags(tag_to_num)

        # format y_preds
        print 'formatting predicted y tags for evaluation...'
        train_predictions, probabilities, vectors = get_crf_proba(data_file, parameters)
        p_tags = []
        y_preds = []
        p_preds = []  # for each byte: {tag: prob, tag2: prob2, ...}
        for train_idx, train_sample in enumerate(train_predictions):
            tag_ids = train_sample.argmax(axis=1)
            y_preds.append(tag_ids)
            train_sample_tags = []
            train_sample_preds = []
            for tag_id in tag_ids:
                train_sample_tags.append(num_to_tags[tag_id])
            if parameters['tag_scheme'] == 'iobes':
                train_sample_tags = iobes_iob(train_sample_tags)
            p_tags.append(train_sample_tags)
            for sample in train_sample:  # train_sample is 2D, sample is 1D
                tag_to_pred = []
                for tag_idx in range(len(sample)):
                    tag_to_pred.append((num_to_tags[tag_idx], sample[tag_idx]))
                train_sample_preds.append(str(tag_to_pred))
            p_preds.append(train_sample_preds)

        if len(p_tags) != len(r_tags) or len(y_preds) != len(y_reals):
            print 'len p_tags', len(p_tags), p_tags[:2]
            print 'len r_tags', len(r_tags), r_tags[:2]
            print 'len y_preds', len(y_preds), y_preds[:2]
            print 'len y_reals', len(y_reals), y_reals[:2]
        assert len(p_tags) == len(r_tags) and len(y_preds) == len(y_reals)

        # clean X_data, p_tags, y_preds, r_tags, y_reals so that there are no duplicate data
        # if entity starts in one sample and ends in another, overwrite earlier sample prediction
        X_data, r_tags, y_reals, p_tags, y_preds, probabilities, vectors = self.combine_overlapping_data(X_data, r_tags, y_reals, p_tags, y_preds, probabilities, vectors, parameters)

        # combine vector to be per sample
        if parameters['get_vectors']:
            combined_vectors = []
            for vector in vectors:  # (# samples, x)
                vector = np.sum(vector, axis=0)
                combined_vectors.append(vector)
            vectors = combined_vectors

        print 'formatting predictions for comparison'
        for i, (y_pred_sample, y_real_sample) in enumerate(zip(y_preds, y_reals)):
            X_sample_nums = []
            for j, (y_pred, y_real) in enumerate(zip(y_pred_sample, y_real_sample)):
                feat_id = str(X_data[i][j][0]).split(',')[0]
                if int(feat_id):  # not the byte for padding
                    X_sample_num = ','.join(X_data[i][j])  # byte features of this sample
                    predictions.append(' '.join([str(X_sample_num), r_tags[i][j], p_tags[i][j]]))
                    # format prediction_probabilities
                    X_sample_nums.append(int(feat_id))
                    count[y_real, y_pred] += 1
            predictions.append('')

            if parameters['get_probs'] or parameters['get_vectors']:
                X_sample_nums = ''.join([chr(x) for x in X_sample_nums])
                X_sample_nums = X_sample_nums.replace('\t', ' ')
                curr_line = [X_sample_nums]
                if parameters['get_probs']:
                    curr_line.append(str(probabilities[i]))
                if parameters['get_vectors']:
                    vector_strs = []
                    for vector in vectors:
                        vector_strs.append(' '.join([str(v) for v in vector]))
                    curr_line.append(vector_strs[i])
                prediction_probabilities.append('\t'.join(curr_line))

        # Write predictions to disk and run CoNLL script externally
        print 'running CoNLL script...'
        eval_id = np.random.randint(1000000, 2000000)
        temp_dir = parameters['temp_dir']
        output_path = os.path.join(temp_dir, "eval.%i.output" % eval_id)
        probs_path = os.path.join(temp_dir, "eval.%i.probs_vectors" % eval_id)
        scores_path = os.path.join(temp_dir, "eval.%i.scores" % eval_id)
        with codecs.open(output_path, 'wb') as f:
            print 'writing...'
            f.write("\n".join(predictions))  # write byte ids instead of bytes, bc utf8
        os.system("%s < %s > %s" % (eval_script, output_path, scores_path))
        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
        print 'eval ID:', str(eval_id)
        for line in eval_lines:
            print line

        # Write probabilities file
        if parameters['get_probs'] or parameters['get_vectors']:
            with codecs.open(probs_path, 'wb') as f:
                print 'writing probabilities...'
                f.write("\n".join(prediction_probabilities) + "\n")

        # Remove temp files
        # os.remove(output_path)
        # os.remove(scores_path)

        # Confusion matrix with accuracy for each tag
        print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            "ID", "NE", "Total",
            *([num_to_tags[i] for i in xrange(n_tags)] + ["Percent"])
        )
        for i in xrange(n_tags):
            print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
                str(i), num_to_tags[i], str(count[i].sum()),
                *([count[i][j] for j in xrange(n_tags)] +
                  ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
            )

        # Global accuracy
        print "%i/%i (%.5f%%)" % (
            count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
        )

        # F1 on all entities
        return float(eval_lines[1].strip().split()[-1])

    def combine_overlapping_data(self, X_data, r_tags, y_reals, p_tags, y_preds, probabilities, vectors, parameters):
        """
        clean X_data, p_tags, y_preds, r_tags, y_reals so that there are no duplicate data
        if entity starts in one sample and ends in another, overwrite earlier sample prediction
        :param X_data:
        :param r_tags:
        :param y_reals:
        :param p_tags:
        :param y_preds:
        :return:
        """
        max_chars_in_sample = parameters['max_chars_in_sample']

        cleaned_X_data, cleaned_p_tags, cleaned_r_tags, cleaned_y_preds, cleaned_y_reals, cleaned_probabilities, cleaned_vectors = [], [], [], [], [], [], []
        curr_X_data = []
        prev_seq = None
        for sample_idx in range(len(X_data)):
            # only get the first (byte) ID for each data point in sample_x_data
            sample_x_data = X_data[sample_idx]
            sample_x_data_bytes = sample_x_data  # [x[0] for x in sample_x_data]
            sample_p_tags = p_tags[sample_idx]
            sample_y_preds = y_preds[sample_idx]
            if parameters['get_probs']:
                sample_probabilities = [probabilities[sample_idx]]  # one probability per sample
            else:
                sample_probabilities = []
            if parameters['get_vectors']:
                sample_vectors = vectors[sample_idx]
            else:
                sample_vectors = np.array([[]])
            if 'input' not in parameters:  # only if we are evaluating model
                sample_r_tags = r_tags[sample_idx]
                sample_y_reals = y_reals[sample_idx]

            if prev_seq:
                int_max_chars_in_sample = int(max_chars_in_sample)
                prev_overlap = prev_seq[int_max_chars_in_sample / 2:]

                sample_overlap = sample_x_data_bytes[:int_max_chars_in_sample / 2]
                if prev_overlap == sample_overlap:
                    curr_X_data.extend(sample_x_data[int_max_chars_in_sample / 2:])

                    if 'input' not in parameters:  # only if we are evaluating model
                        if sample_r_tags[int_max_chars_in_sample / 2].startswith('I-'):
                            # go backwards until we find a B- tag. replace old assigned tag with new pred
                            i = int_max_chars_in_sample / 2
                            tag = sample_r_tags[i]
                            backtracked = 0
                            while not tag.startswith('B-'):
                                i -= 1
                                tag = sample_r_tags[i]
                                backtracked += 1
                            curr_r_tags = curr_r_tags[:-backtracked]
                            curr_r_tags.extend(sample_r_tags[i:])
                            curr_y_reals = curr_y_reals[:-backtracked]
                            curr_y_reals.extend(sample_y_reals[i:])
                        else:
                            curr_r_tags.extend(sample_r_tags[int_max_chars_in_sample / 2:])
                            curr_y_reals.extend(sample_y_reals[int_max_chars_in_sample / 2:])

                    if sample_p_tags[int_max_chars_in_sample / 2].startswith('I-'):
                        # go backwards until we find a B- tag. replace old assigned tag with new pred
                        # predictions might be malformatted, but it's ok. conlleval script will take care of it.
                        i = int_max_chars_in_sample / 2
                        tag = sample_p_tags[i]
                        tag_split = tag.split('-')
                        curr_entity = tag_split[1]
                        entity = curr_entity
                        backtracked = 0
                        while i > 0 and tag.startswith('I-') and curr_entity == entity:
                            i -= 1
                            tag = sample_p_tags[i]
                            entity = ''
                            tag_split = tag.split('-')
                            if len(tag_split) > 1:
                                entity = tag_split[1]
                            backtracked += 1
                        curr_p_tags = curr_p_tags[:-backtracked]
                        curr_p_tags.extend(sample_p_tags[i:])
                        curr_y_preds = curr_y_preds[:-backtracked]
                        curr_y_preds.extend(sample_y_preds[i:])
                        curr_vectors = curr_vectors[:-backtracked, :]
                        curr_vectors = np.concatenate((curr_vectors, sample_vectors[i:, :]), axis=0)
                    else:
                        curr_p_tags.extend(sample_p_tags[int_max_chars_in_sample / 2:])
                        curr_y_preds.extend(sample_y_preds[int_max_chars_in_sample / 2:])
                        curr_vectors = np.concatenate((curr_vectors, sample_vectors[int_max_chars_in_sample / 2:, :]), axis=0)

                    curr_probabilities.extend(sample_probabilities)

                else:  # end of a sample, beginning a new sample
                    cleaned_X_data.append(curr_X_data)
                    cleaned_p_tags.append(curr_p_tags)
                    cleaned_y_preds.append(curr_y_preds)
                    cleaned_probabilities.append(np.mean(curr_probabilities))
                    cleaned_vectors.append(curr_vectors)
                    curr_X_data = list(sample_x_data)
                    curr_p_tags = list(sample_p_tags)
                    curr_y_preds = list(sample_y_preds)
                    curr_probabilities = list(sample_probabilities)
                    curr_vectors = np.array(sample_vectors)
                    if 'input' not in parameters:  # only if we are evaluating model
                        cleaned_r_tags.append(curr_r_tags)
                        cleaned_y_reals.append(curr_y_reals)
                        curr_r_tags = list(sample_r_tags)
                        curr_y_reals = list(sample_y_reals)

            else:  # first sample
                curr_X_data = list(sample_x_data)
                curr_p_tags = list(sample_p_tags)
                curr_y_preds = list(sample_y_preds)
                curr_probabilities = list(sample_probabilities)
                curr_vectors = np.array(sample_vectors)
                if 'input' not in parameters:  # only if we are evaluating model
                    curr_r_tags = list(sample_r_tags)
                    curr_y_reals = list(sample_y_reals)

            prev_seq = sample_x_data_bytes

        if len(curr_X_data) != 0:
            cleaned_X_data.append(curr_X_data)
            cleaned_p_tags.append(curr_p_tags)
            cleaned_y_preds.append(curr_y_preds)
            cleaned_probabilities.append(np.mean(curr_probabilities))
            cleaned_vectors.append(curr_vectors)
            if 'input' not in parameters:  # only if we are evaluating model
                cleaned_r_tags.append(curr_r_tags)
                cleaned_y_reals.append(curr_y_reals)

        return cleaned_X_data, cleaned_r_tags, cleaned_y_reals, cleaned_p_tags, cleaned_y_preds, cleaned_probabilities, cleaned_vectors


def convert_enc_dec_output_to_IOB(input_seqs, decoded_seqs, params):
    """
    Convert output of encoder-decoder model [S0, L3, entity1,....<STOP>] into IOB format
    :param input_seqs:
    :param decoded_seqs:
    :return:
    """
    tags = params['tags']
    iob_output = []
    for input_seq, decoded_seq in zip(input_seqs, decoded_seqs):  # one batch of input_seqs
        output_seq = []

        # get rid of malformed predictions in decoded_seq
        decoded_idx = 0
        processed_decoded_seq = []
        while decoded_idx + 2 < len(decoded_seq):
            if decoded_seq[decoded_idx] == '<STOP>':
                break
            s = decoded_seq[decoded_idx]
            l = decoded_seq[decoded_idx + 1]
            e = decoded_seq[decoded_idx + 2]
            if s.startswith('S') and l.startswith('L') and e in tags:
                try:
                    int_s = int(s[1:])
                    int_l = int(l[1:])
                    if int_s + int_l <= len(input_seq):  # valid
                        processed_decoded_seq.append((int_s, int_l, e))
                    decoded_idx += 3
                except ValueError:
                    decoded_idx += 1
            else:
                decoded_idx += 1

        # sort decoded_seq based on offset from greatest to least
        ordered_decoded_seq = reversed(sorted(processed_decoded_seq, key=lambda x: x[0]))

        # for overlapping predictions, keep the one with a greater offset (because we're formatting the groundtruth output backwards, so the model probably does better at predicting larger offsets)
        seen_offsets = {}  # dictionary of offsets to (length, entity)
        annotated_offsets = set()
        for offset, entity_length, entity in ordered_decoded_seq:
            annotation_exists = False
            for i in range(offset, offset + entity_length):
                if i in annotated_offsets:  # skip to the next annotation
                    annotation_exists = True
                    break
            if not annotation_exists:  # add annotation
                seen_offsets[offset] = (entity_length, entity)
                for i in range(offset, offset + entity_length):
                    annotated_offsets.add(i)

        # populate iob_output
        byt_idx = 0
        while byt_idx < len(input_seq):
            if byt_idx in seen_offsets:
                entity_length = seen_offsets[byt_idx][0]
                entity = seen_offsets[byt_idx][1]
                output_seq.append('B-' + entity)
                entity_length -= 1
                byt_idx += 1
                while entity_length > 0:
                    output_seq.append('I-' + entity)
                    entity_length -= 1
                    byt_idx += 1
            else:
                output_seq.append('O')
                byt_idx += 1

        iob_output.append(output_seq)

    return iob_output


def collect_tags(data_file):
    """
    Collect entity tags we want to predict from data_file
    :param data_file:
    :return:
    """
    tags = []
    with open(data_file, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                for sample_batch in data:
                    annotation_list = sample_batch[1]
                    for sentence in annotation_list:
                        for ann in sentence:
                            entity = ann[2]
                            if entity not in tags:
                                tags.append(entity)
            except EOFError as _:
                break
    return tags


def remove_duplicate_samples(data_file):
    """
    Remove duplicate samples from data_file.src and data_file.tgt
    :param data_file:
    :return:
    """
    src_file = data_file + '.src'
    tgt_file = data_file + '.tgt'
    seen_samples = OrderedDict()
    with open(src_file, 'rb') as s, open(tgt_file, 'rb') as t:
        for s_line, t_line in zip(s, t):
            if s_line not in seen_samples:
                seen_samples[s_line] = t_line

    new_src_file = data_file + '.unique.src'
    new_tgt_file = data_file + '.unique.tgt'
    with open(new_src_file, 'wb') as ns, open(new_tgt_file, 'wb') as nt:
        for s_line in seen_samples:
            t_line = seen_samples[s_line]
            ns.write(s_line)
            nt.write(t_line)

    return data_file + '.unique'
