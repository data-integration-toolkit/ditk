# -*- coding: utf-8 -*-
"""
Preprocessing code to clean and convert raw data from BioCreative files to format usable by models
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
from collections import OrderedDict
from xml.etree import ElementTree
import os
import random
import io
import cPickle as pickle
from itertools import izip
import codecs
import re


annotations_typ_map = {'Uniprot': 'proteingene',
                       'protein': 'proteingene',
                       'NCBI gene': 'proteingene',
                       'gene': 'proteingene',
                       'CHEBI': 'smallmolecule',
                       'PubChem': 'smallmolecule',
                       'molecule': 'smallmolecule',
                       'GO': 'cellularcomponent',
                       'subcellular': 'cellularcomponent',
                       'CVCL': 'celltypeline',
                       'CL': 'celltypeline',
                       'cell': 'celltypeline',
                       'Uberon': 'tissuesorgans',
                       'tissue': 'tissuesorgans',
                       'NCBI taxon': 'organismsspecies',
                       'organism': 'organismsspecies'}


def preprocess_biocreative(ann_folder, format='iob'):  # format = [iob|encdec|st]
    """
    Preprocess biocreative data
    :param ann_folder:
    :return:
    """
    for ann_file in os.listdir(ann_folder):
        all_sentences = OrderedDict()  # { pmc_id : sentences_str }
        all_annotations = OrderedDict()  # {pmc_id : sentences_annotations_list }
        print 'file:', ann_file
        output_file = os.path.join(ann_folder, ann_file) + '.char-iob'
        if format == 'encdec':
            output_file += '.encdec'

        if ann_file.endswith('.xml'):
            root = ElementTree.parse(os.path.join(ann_folder, ann_file)).getroot()
            # root -> document -> passage -> text -> text
            #                             -> annotation -> location
            #                                           -> text -> text
            #                                           ->  infon key="type" (type of entity)
            #                  -> id
            documents = root.findall('document')
            for document in documents:
                id = document.find('id').text
                id = id.replace(' ', '|')
                passages = document.findall('passage')
                for passage in passages:
                    text = passage.find('text')  # should be one per passage
                    annotations = passage.findall('annotation')
                    sentence = text.text
                    if '\xe2\x80\xa8' in sentence:
                        print 'id', id
                        print sentence
                        sentence = sentence.replace('\xe2\x80\xa8', '')
                        print sentence
                    all_sentences[id] = sentence
                    curr_annotations = []
                    seen_offset_lens = {}  # in case some entities are multiply annotated
                    for ann in annotations:
                        loc = ann.find('location')
                        loc_offset = loc.attrib['offset']
                        loc_length = loc.attrib['length']
                        loc_text = ann.find('text').text
                        typ = ann.find('.//infon[@key="type"]').text
                        print typ
                        typ = typ.split('_')[0].split(':')[0]
                        not_types = ['BAO', 'Rfam', 'Corum']
                        if typ in not_types:
                            pass
                        else:
                            typ = annotations_typ_map[typ]
                            if not (loc_offset in seen_offset_lens and seen_offset_lens[loc_offset] == loc_length):
                                curr_annotations.append((loc_text, loc_offset, loc_length, typ))
                                seen_offset_lens[loc_offset] = loc_length
                    all_annotations[id] = curr_annotations

            # format annotations
            tok_annotations = OrderedDict()
            for id in all_annotations:
                if id not in tok_annotations:
                    tok_annotations[id] = OrderedDict()
                for curr_text, off, length, typ in all_annotations[id]:
                    tok_annotations[id][int(off)] = (curr_text, typ, int(length))
                tok_annotations[id] = OrderedDict(sorted(tok_annotations[id].items()))

            # output tokenized sentences
            if format == 'encdec':
                write_enc_dec_format(all_sentences, tok_annotations, output_file)
            elif format == 'iob':
                write_iob_format(all_sentences, tok_annotations, output_file)
            elif format == 'st':
                write_src_tgt_format(all_sentences, tok_annotations, output_file)
            else:
                print 'Unknown output format'
                exit()

        else:
            print 'skipping...'


def make_src_tgt_from_bpe_and_iob(bpe_file, iob_file, src_file, tgt_file):
    """
    Make src and tgt files from sentence file (bpe file) and iob file
    :param bpe_file:
    :param iob_file:
    :return:
    """
    with open(bpe_file, 'rb') as opened_bpe, open(iob_file, 'rb') as opened_iob:
        with open(src_file, 'wb') as opened_src, open(tgt_file, 'wb') as opened_tgt:
            while True:
                try:
                    bpe_data = pickle.load(opened_bpe)
                    iob_data = pickle.load(opened_iob)
                    for b in bpe_data:
                        opened_src.write(b + '\n')
                    for i in iob_data:
                        tag_idx = 0
                        curr_tgts = []
                        while tag_idx < len(i):
                            tag = i[tag_idx]
                            if tag.startswith('B-'):
                                curr_start = tag_idx
                                entity = tag[2:]
                                tag_idx += 1
                                if tag_idx < len(i):
                                    tag = i[tag_idx]
                                    while tag.startswith('I-') and tag_idx < len(i):
                                        tag_idx += 1
                                        if tag_idx < len(i):
                                            tag = i[tag_idx]
                                curr_tgts.insert(0, entity)
                                curr_tgts.insert(0, 'L' + str(tag_idx - curr_start))
                                curr_tgts.insert(0, 'S' + str(curr_start))
                            else:
                                # should only be O tags
                                tag_idx += 1
                        opened_tgt.write(' '.join(curr_tgts) + '\n')
                except EOFError:
                    break


def split_sentences_from_src_tgt(src_file, tgt_file):
    """
    Split each line in src/tgt file into approximately one sentence per line
    :param src_file:
    :param tgt_file:
    :return:
    """
    src_file_output = src_file + '.sentences'
    tgt_file_output = tgt_file + '.sentences'
    with open(src_file, 'rb') as s, open(tgt_file, 'rb') as t, open(src_file_output, 'wb') as so, open(tgt_file_output, 'wb') as to:
        for src_line, tgt_line in zip(s, t):
            src_line = src_line.rstrip()
            tgt_line = tgt_line.strip()
            sentence_annotations = tgt_line.split()
            # split sentences on "<lowercase>. " patterns
            match = re.finditer(r'[^A-Z]\. [^a-z]', src_line)
            curr_start = 0
            for m in match:
                start = m.start(0)
                end = m.end(0)
                curr_end = start + 2
                curr_sent = src_line[curr_start:curr_end]
                so.write(curr_sent + '\n')

                # re-calculate tgts
                tgt_idx = 0
                curr_tgts = []
                while tgt_idx < len(sentence_annotations):
                    offset = sentence_annotations[tgt_idx]
                    entity_length = sentence_annotations[tgt_idx + 1]
                    entity = sentence_annotations[tgt_idx + 2]
                    # "S89" -> 89
                    offset = int(offset[1:])
                    entity_length = int(entity_length[1:])
                    tgt_idx += 3

                    if offset < curr_start:  # annotated entity ends before current sample range
                        pass  # break
                    elif offset >= curr_end:  # annotated entity begins after current sample range
                        pass  # annotated entities are sorted in reverse by offset
                    else:  # annotated entity in current sample range
                        if offset + entity_length - 1 < curr_end:
                            curr_tgts.append('S' + str(offset - curr_start))
                            curr_tgts.append('L' + str(entity_length))
                            curr_tgts.append(entity)
                        else:
                            pass
                to.write(' '.join(curr_tgts) + '\n')

                curr_start = end - 1

            # last sentence in paragraph
            curr_end = len(src_line)
            curr_sent = src_line[curr_start:curr_end]
            so.write(curr_sent + '\n')

            # re-calculate tgts
            tgt_idx = 0
            curr_tgts = []
            while tgt_idx < len(sentence_annotations):
                offset = sentence_annotations[tgt_idx]
                entity_length = sentence_annotations[tgt_idx + 1]
                entity = sentence_annotations[tgt_idx + 2]
                # "S89" -> 89
                offset = int(offset[1:])
                entity_length = int(entity_length[1:])
                tgt_idx += 3

                if offset < curr_start:  # annotated entity ends before current sample range
                    pass  # break
                elif offset >= curr_end:  # annotated entity begins after current sample range
                    pass  # annotated entities are sorted in reverse by offset
                else:  # annotated entity in current sample range
                    if offset + entity_length - 1 < curr_end:
                        curr_tgts.append('S' + str(offset - curr_start))
                        curr_tgts.append('L' + str(entity_length))
                        curr_tgts.append(entity)
                    else:
                        pass
            to.write(' '.join(curr_tgts) + '\n')


def write_iob_format(tok_sentences, tok_annotations, output_file):
    """
    Write preprocessed data in IOB format (for byte-level characters).
    :param tok_sentences:
    :param tok_annotations:
    :param output_file:
    :return:
    """
    data = []
    with io.open(output_file, 'wb') as o:
        ids = tok_sentences.keys()
        for idx, id in enumerate(ids):
            annotations = tok_annotations[id]
            sentences = bytes.encode(str(tok_sentences[id]))
            char_idx = 0
            while char_idx < len(sentences):
                char = sentences[char_idx]
                if ord(char) < 0 or ord(char) > 255:
                    print 'CHAR', char
                    exit()

                if char == ' ':
                    char = '<SPACE>'

                if char_idx in annotations:
                    ann_info = annotations[char_idx]
                    typ = ann_info[1]
                    new_typ = typ.split(':')[0].split('_')[0]
                    length = ann_info[2]
                    curr_ent_len = 0
                    while curr_ent_len < length:
                        char = sentences[char_idx]

                        if char == ' ':
                            char = '<SPACE>'

                        if new_typ in annotations_typ_map:
                            mapped_typ = annotations_typ_map[new_typ]
                            if curr_ent_len == 0:
                                data.append(char + ' B-' + mapped_typ)
                            else:
                                data.append(char + ' I-' + mapped_typ)
                        else:
                            data.append(char + ' O')

                        curr_ent_len += 1
                        char_idx += 1
                else:
                    data.append(char + ' O')
                    char_idx += 1

            # put spaces between paragraph snippets if we're going to concat the snippets
            if idx != len(ids) - 1:
                data.append('<SPACE> O')

        pickle.dump(data, o)


def split_train_dev_test_sentences(doc_folder, train_file, dev_file, test_file, ext):
    """
    Split sentence data (unpickled) into train, dev, test.
    :param doc_folder:
    :param train_file:
    :param dev_file:
    :param test_file:
    :param ext:
    :return:
    """
    random.seed(123)
    # order could be different if the files are called different things?
    files = [f for f in os.listdir(doc_folder) if f.endswith(ext)]
    src_files = [f + '.src' for f in files]
    tgt_files = [f + '.tgt' for f in files]

    data_len = len(files)
    train_threshold = int(0.80 * data_len)
    dev_threshold = train_threshold + int(0.10 * data_len)
    sample = random.sample([x for x in range(data_len)], data_len)

    if dev_file == None and test_file == None:  # write everything to "train_file"
        with open(train_file + '.src', 'wb') as train_s, open(train_file + '.tgt', 'wb') as train_t:
            for i in range(data_len):
                s_file = src_files[i]
                t_file = tgt_files[i]
                with open(os.path.join(doc_folder, s_file), 'rb') as fi:
                    train_s.write(fi.read())
                with open(os.path.join(doc_folder, t_file), 'rb') as fi:
                    train_t.write(fi.read())
    elif test_file == None:  # write everything to train_file and dev_file
        train_threshold = int(0.9 * data_len)
        with open(train_file + '.src', 'wb') as train_s, open(train_file + '.tgt', 'wb') as train_t:
            with open(dev_file + '.src', 'wb') as dev_s, open(dev_file + '.tgt', 'wb') as dev_t:
                for i in range(data_len):
                    s_file = src_files[i]
                    t_file = tgt_files[i]
                    with open(os.path.join(doc_folder, s_file), 'rb') as fi:
                        s_lines = fi.read()
                    with open(os.path.join(doc_folder, t_file), 'rb') as fi:
                        t_lines = fi.read()
                    if sample[i] < train_threshold:
                        train_s.write(s_lines)
                        train_t.write(t_lines)
                    else:
                        dev_s.write(s_lines)
                        dev_t.write(t_lines)
    else:
        with open(train_file + '.src', 'wb') as train_s, open(train_file + '.tgt', 'wb') as train_t:
            with open(dev_file + '.src', 'wb') as dev_s, open(dev_file + '.tgt', 'wb') as dev_t:
                with open(test_file + '.src', 'wb') as test_s, open(test_file + '.tgt', 'wb') as test_t:
                    for i in range(data_len):
                        s_file = src_files[i]
                        t_file = tgt_files[i]
                        with open(os.path.join(doc_folder, s_file), 'rb') as fi:
                            s_lines = fi.read()
                        with open(os.path.join(doc_folder, t_file), 'rb') as fi:
                            t_lines = fi.read()
                        if sample[i] < train_threshold:
                            train_s.write(s_lines)
                            train_t.write(t_lines)
                        elif sample[i] < dev_threshold:
                            dev_s.write(s_lines)
                            dev_t.write(t_lines)
                        else:
                            test_s.write(s_lines)
                            test_t.write(t_lines)


def split_train_dev_test(doc_folder, train_file, dev_file, test_file, ext):
    """
    Split data into train, dev, test.
    :param ann_folder:
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    """
    random.seed(123)
    files = [f for f in os.listdir(doc_folder) if f.endswith(ext)]
    data_len = len(files)
    train_threshold = int(0.80 * data_len)
    dev_threshold = train_threshold + int(0.10 * data_len)
    sample = random.sample([x for x in range(data_len)], data_len)

    if dev_file == None and test_file == None:  # write everything to "train_file"
        train_arr = []
        with open(train_file, 'wb') as train:
            for i in range(data_len):
                filer = files[i]
                with open(os.path.join(doc_folder, filer), 'rb') as fi:
                    lines = pickle.load(fi)
                train_arr.append(lines)
            pickle.dump(train_arr, train)
    elif test_file == None:  # write everything to train_file and dev_file
        train_threshold = int(0.9 * data_len)
        train_arr = []
        dev_arr = []
        with open(train_file, 'wb') as train:
            with open(dev_file, 'wb') as dev:
                for i in range(data_len):
                    filer = files[i]
                    with open(os.path.join(doc_folder, filer), 'rb') as fi:
                        lines = pickle.load(fi)
                    if sample[i] < train_threshold:
                        train_arr.append(lines)
                    else:
                        dev_arr.append(lines)
                pickle.dump(train_arr, train)
                pickle.dump(dev_arr, dev)
    else:
        train_arr = []
        dev_arr = []
        test_arr = []
        with open(train_file, 'wb') as train:
            with open(dev_file, 'wb') as dev:
                with open(test_file, 'wb') as test:
                    for i in range(data_len):
                        filer = files[i]
                        with open(os.path.join(doc_folder, filer), 'rb') as fi:
                            lines = pickle.load(fi)
                        if sample[i] < train_threshold:
                            train_arr.append(lines)
                        elif sample[i] < dev_threshold:
                            dev_arr.append(lines)
                        else:
                            test_arr.append(lines)
                    pickle.dump(train_arr, train)
                    pickle.dump(dev_arr, dev)
                    pickle.dump(test_arr, test)


def write_src_tgt_format(tok_sentences, tok_annotations, output_file):
    """
    Write src and tgt files.
    :param tok_sentences:
    :param tok_annotations:
    :param output_file:
    :return:
    """
    new_annotations = []
    byts = []
    src_file = output_file + '.src'
    tgt_file = output_file + '.tgt'

    with io.open(src_file, 'wb') as s, io.open(tgt_file, 'wb') as t:
        ids = tok_sentences.keys()
        for idx, id in enumerate(ids):
            annotations = tok_annotations[id]
            sent_annotations = []
            sentences = tok_sentences[id]
            sentences = bytes.encode(str(sentences))
            sent_bytes = [byt for byt in sentences]
            byts.append(''.join(sent_bytes))

            for offset in annotations:
                ann_info = annotations[offset]
                typ = ann_info[1]
                length = ann_info[2]
                sent_annotations.append(('S' + str(offset), 'L' + str(length), typ))
            sent_annotations = list(reversed(sent_annotations))
            # flatten annotations
            temp_sent_annotations = []
            for o,l,e in sent_annotations:
                temp_sent_annotations.append(o)
                temp_sent_annotations.append(l)
                temp_sent_annotations.append(e)
            sent_annotations = temp_sent_annotations
            new_annotations.append(' '.join(sent_annotations))
            # print sent_annotations
        s.write('\n'.join(byts) + '\n')
        t.write('\n'.join(new_annotations) + '\n')


def write_enc_dec_format(tok_sentences, tok_annotations, output_file):
    """
    Write (offset, len, type) annotations for each sentence, one sentence per line
    :param all_annotations:
    :param output_file:
    :return:
    """
    new_annotations = []
    byts = []

    with io.open(output_file, 'wb') as o:
        ids = tok_sentences.keys()
        for idx, id in enumerate(ids):
            annotations = tok_annotations[id]
            sent_annotations = []
            sentences = tok_sentences[id]
            sentences = bytes.encode(str(sentences))
            sent_bytes = [byt for byt in sentences]
            byts.append(sent_bytes)
            #print sent_bytes

            for offset in annotations:
                ann_info = annotations[offset]
                typ = ann_info[1]
                length = ann_info[2]
                sent_annotations.append((offset, length, typ))
            sent_annotations = list(reversed(sent_annotations))
            new_annotations.append(sent_annotations)
            #print sent_annotations
        pickle.dump((byts, new_annotations), o)


def write_user_input_to_model_input(input_src_file, input_tgt_file, output_file):
    """
    Convert from user input format (file of sentences, file of tuples of annotations)
    to model input file format (pickle file of byte list and list of annotations)
    :param input_file:
    :param output_file:
    :return:
    """
    new_annotations = []
    byts = []

    if len(input_tgt_file) > 0:  # in training mode
        with codecs.open(input_src_file, 'rb') as src, codecs.open(input_tgt_file, 'rb') as tgt:
            for src_line, tgt_line in izip(src, tgt):
                src_line = src_line[:-1]  # strip last newline
                tgt_line = tgt_line

                # convert annotations
                t_idx = 0
                tgt_line_split = tgt_line.strip().split()
                sent_annotations = []
                while t_idx < len(tgt_line_split):
                    offset = tgt_line_split[t_idx]
                    length = tgt_line_split[t_idx + 1]
                    entity = tgt_line_split[t_idx + 2]
                    sent_annotations.append((offset, length, entity))
                    t_idx += 3
                new_annotations.append(sent_annotations)

                # convert bytes
                sent_bytes = [byt for byt in src_line]
                byts.append(sent_bytes)
    else:
        with codecs.open(input_src_file, 'rb') as src:
            for src_line in src:
                src_line = src_line[:-1]
                new_annotations.append([])

                # convert bytes
                sent_bytes = [byt for byt in src_line]
                byts.append(sent_bytes)

    with open(output_file, 'wb') as o:
        pickle.dump([(byts, new_annotations)], o)  # need the extra [] for backwards compatibility with model input


def model_input_to_user_input(input_file, output_src_file, output_tgt_file):
    """
    Convert from model input format (pickle file of byte list and list of annotations)
    to user input format (file of sentences, file of tupes of annotations)
    :param input_file:
    :param output_src_file:
    :param output_tgt_file:
    :return:
    """
    srces = []
    tgts = []
    with open(input_file, 'rb') as i:
        data = pickle.load(i)
        for data_tuple in data:
            byts = data_tuple[0]
            annotations = data_tuple[1]
            for b,a in zip(byts, annotations):
                sent = ''.join(b)
                new_annotations = []
                for o,l,e in a:
                    new_annotations.append('S' + str(o))
                    new_annotations.append('L' + str(l))
                    new_annotations.append(str(e))
                sent = sent.replace('\n', ' ')
                srces.append(sent)
                tgts.append(' '.join(new_annotations))
    with open(output_src_file, 'wb') as s, open(output_tgt_file, 'wb') as t:
        s.write('\n'.join(srces) + '\n')
        t.write('\n'.join(tgts) + '\n')


def write_user_byte_iob_input_to_src_tgt_input(input_file, output_src_file, output_tgt_file, space_token='<SPACE>'):
    """
    Write user byte IOB input (IOB file at the byte level)
    to src file (one sentence per line) and tgt file (annotations per line)
    :param input_file:
    :param output_src_file:
    :param output_tgt_file:
    :return:
    """
    # keep track of current sentence and annotations
    sentence = []
    ann = []
    curr_ann_idx = 0
    curr_ann = []

    # all sentences and annotations
    sentences = []
    anns = []

    with codecs.open(input_file, 'rb') as iob_file:
        for line in iob_file:
            line = line.strip()

            if len(line) > 0:  # same sentence
                split_line = line.split(' ')
                byt = split_line[0]
                # replace space_token with actual space
                if byt == space_token:
                    byt = ' '
                sentence.append(byt)
                tag = split_line[-1]
                if tag.startswith('B-'):
                    if len(curr_ann) == 2:  # insert length and type of prev entity
                        curr_ann = ('S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1]), curr_ann[0])
                        ann.insert(0, ' '.join(curr_ann))
                        curr_ann = []
                    tag_type = tag[2:]
                    curr_ann.append(tag_type)
                    curr_ann.append(curr_ann_idx)
                elif tag == 'O':
                    if len(curr_ann) == 2:  # insert length and type of prev entity
                        curr_ann = ['S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1]), curr_ann[0]]
                        ann.insert(0, ' '.join(curr_ann))
                        curr_ann = []
                else:
                    if len(curr_ann) != 2:
                        print 'seeing stray I- tag!'
                        exit()
                curr_ann_idx += 1
            else:  # new sentence
                if len(curr_ann) == 2:  # insert length and type of prev entity
                    curr_ann = ('S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1]), curr_ann[0])
                    ann.insert(0, ' '.join(curr_ann))
                    curr_ann = []
                # save old sentence
                if len(sentence) > 0:
                    sentences.append(''.join(sentence))
                    anns.append(' '.join(ann))
                sentence = []
                ann = []
                curr_ann_idx = 0

        if len(sentence) > 0:  # last sentence and annotations
            sentences.append(''.join(sentence))
            anns.append(' '.join(ann))

    with codecs.open(output_src_file, 'wb') as s, codecs.open(output_tgt_file, 'wb') as t:
        s.write('\n'.join(sentences) + '\n')
        t.write('\n'.join(anns) + '\n')


def write_user_word_iob_input_to_src_tgt_input(input_file, output_src_file, output_tgt_file):
    """
    Write user word IOB input (IOB file at the word level)
    to src file (one sentence per line) and tgt file (annotations per line)
    :param input_file:
    :param output_src_file:
    :param output_tgt_file:
    :return:
    """
    # keep track of current sentence and annotations
    sentence = []
    ann = []
    curr_ann_idx = 0
    curr_ann = []

    # all sentences and annotations
    sentences = []
    anns = []

    with codecs.open(input_file, 'rb') as iob_file:
        for line in iob_file:
            line = line.strip()

            if len(line) > 0:  # same sentence
                split_line = line.split()
                word = split_line[0]
                sentence.append(word)
                tag = split_line[-1]
                if tag.startswith('B-'):
                    if len(curr_ann) == 2:  # insert length and type of prev entity
                        curr_ann = ('S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1] - 1), curr_ann[0])
                        ann.insert(0, ' '.join(curr_ann))
                        curr_ann = []
                    tag_type = tag[2:]
                    curr_ann.append(tag_type)
                    curr_ann.append(curr_ann_idx)
                elif tag == 'O':
                    if len(curr_ann) == 2:  # insert length and type of prev entity
                        curr_ann = ['S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1] - 1), curr_ann[0]]
                        ann.insert(0, ' '.join(curr_ann))
                        curr_ann = []
                else:
                    if len(curr_ann) != 2:
                        print 'seeing stray I- tag!'
                        exit()
                curr_ann_idx += len(word) + 1  # 1 for space
            else:  # new sentence
                if len(curr_ann) == 2:  # insert length and type of prev entity
                    curr_ann = ('S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1] - 1), curr_ann[0])
                    ann.insert(0, ' '.join(curr_ann))
                    curr_ann = []
                # save old sentence
                if len(sentence) > 0:
                    sentences.append(' '.join(sentence))
                    anns.append(' '.join(ann))
                sentence = []
                ann = []
                curr_ann_idx = 0

        if len(sentence) > 0:  # last sentence and annotations
            sentences.append(' '.join(sentence))
            anns.append(' '.join(ann))

    with codecs.open(output_src_file, 'wb') as s, codecs.open(output_tgt_file, 'wb') as t:
        s.write('\n'.join(sentences) + '\n')
        t.write('\n'.join(anns) + '\n')


def convert_src_tgt_to_byte_iob(src_file, tgt_file, byte_iob_file):
    """
    Convert src and tgt files to byte IOB file
    :param src_file:
    :param tgt_file:
    :param byte_iob_file:
    :return:
    """
    with open(src_file, 'rb') as s, open(tgt_file, 'rb') as t, open(byte_iob_file, 'wb') as w:

        for s_line, t_line in zip(s, t):
            s_line = s_line.rstrip('\n')
            # correlate byte annotations to word annotations
            tgt_line_split = t_line.strip().split()
            t_idx = 0
            byte_offsets_to_ann = {}
            while t_idx < len(tgt_line_split):
                offset = int(tgt_line_split[t_idx][1:])
                length = int(tgt_line_split[t_idx + 1][1:])
                entity = tgt_line_split[t_idx + 2]
                t_idx += 3
                byte_offsets_to_ann[offset] = (length, entity)

            # iterate through words in s_line, saving to iob_array
            byte_idx = 0
            while byte_idx < len(s_line):
                if byte_idx in byte_offsets_to_ann:
                    length, entity = byte_offsets_to_ann[byte_idx]
                    print(length, entity, byte_idx)
                    for i in range(length):
                        byt = s_line[byte_idx]
                        if byt == ' ':
                            byt = '<SPACE>'
                        if i == 0:
                            w.write(byt + ' B-' + entity + '\n')
                        else:
                            w.write(byt + ' I-' + entity + '\n')
                        byte_idx += 1
                else:
                    byt = s_line[byte_idx]
                    if byt == ' ':
                        byt = '<SPACE>'
                    w.write(byt + ' ' + 'O\n')
                    byte_idx += 1

            w.write('\n')


def sanity_check_output(input_file):
    with open(input_file, 'rb') as f:
        all = []
        for line in f:
            line = line.strip()
            if len(line) > 0:
                line_split = line.split()
                tok_num = line_split[0].split(',')[0]
                all.append(chr(int(tok_num)))
            else:
                if len(all) > 0:
                    print ''.join(all)
                all = []
        if len(all) > 0:
            print ''.join(all)


def combine_ground_and_tagged_iob(ground_file, tagged_file):
    """
    Combine ground truth and tagged byte IOB formats.
    :param ground_file:
    :param tagged_file:
    :return:
    """
    output_file = tagged_file + '.conll'
    with open(ground_file, 'rb') as g, open(tagged_file, 'rb') as t, open(output_file, 'wb') as o:
        for g_line, t_line in zip(g, t):
            g_line = g_line.strip()
            t_line = t_line.strip()
            if len(g_line) > 0 and len(t_line) > 0:
                g_line_split = g_line.split()
                t_line_split = t_line.split()
                byt = g_line_split[0]
                ground = g_line_split[1]
                tag = t_line_split[-1]
                o.write(byt + ' ' + ground + ' ' + tag + '\n')
            else:
                o.write('\n')


def nertok_to_char_iob_features(nertok_file):
    """
    Convert from NERSuite tokenizer format to byte feature IOB format. (IOB denoting tokenization)
    Tokenizer format:
        ...	...	...
        8	10	of
        11	14	ZAP
        14	15	-
    :param nertok_file:
    :return:
    """
    curr_offset = 0
    tgt_data = []
    curr_tgt_data = []
    orig_end = 0
    with open(nertok_file, 'rb') as n:
        n = n.readlines()

    for line_idx, line in enumerate(n):
        line = line.strip()
        if len(line) > 0:
            split = line.split('\t')
            start = int(split[0]) - curr_offset
            orig_end = int(split[1])
            end = orig_end - curr_offset

            if start < 0:
                print('start is negative', start, end)
                exit()

            curr_tgt_data.insert(0, 'TOK')  # placeholder for consistent src/tgt format as NER entities
            curr_tgt_data.insert(0, 'L' + str(end - start))
            curr_tgt_data.insert(0, 'S' + str(start))
        else:
            curr_offset = orig_end + 1
            if len(curr_tgt_data) > 0:
                tgt_data.append(' '.join(curr_tgt_data))
                curr_tgt_data = []

    if len(curr_tgt_data) > 0:
        tgt_data.append(' '.join(curr_tgt_data))

    output_iob_file = nertok_file + '.tgt'
    with open(output_iob_file, 'wb') as o:
        o.write('\n'.join(tgt_data) + '\n')
