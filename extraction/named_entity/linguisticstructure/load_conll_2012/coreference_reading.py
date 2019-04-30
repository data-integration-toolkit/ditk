#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import load_conll_2012.pstree as pstree, load_conll_2012.treebanks as treebanks, load_conll_2012.head_finder as head_finder
from collections import defaultdict
from io import StringIO
import re
import codecs


def get_parse_spans(parses, word ,word_index):
    if len(parses.subtrees) > 0:
        for parse in parses.subtrees:
            index = get_parse_spans(parse, word ,word_index)
            if index:
                return index
    else:
        if parses.word == word and parses.span[0] == word_index[0] and parses.span[1] == word_index[1]:
            return parses.parent.span
    pass

def read_conll_parses(lines):
    in_file = StringIO(''.join(lines))
    return treebanks.read_trees(in_file, treebanks.conll_read_tree)

def read_conll_text(lines):
    """
        Get the words from text corpora
    """
    text = [[]]
    for line in lines:
        line = line.strip()
        fields = re.split(r'\s+', line)
        if len(line) == 0:
            text.append([])
        else:
            text[-1].append(fields[3])
    if len(text[-1]) == 0:
        text.pop()
    return text

def read_conll_ner(lines):
    info = {}
    word = 0
    sentence = 0
    cur = []
    for line in lines:
        line = line.strip()
        fields = re.split(r'\s+', line)
        if len(fields) >= 11:
            ner_info = fields[10]
            if '(' in ner_info and '*' in ner_info:
                cur.append((ner_info[1:-1], sentence, word))
            elif '(' in ner_info and ')' in ner_info:
                info[sentence, word, word +1] = ner_info[1:-1]
            elif ')' in ner_info and '*' in ner_info:
                start = cur.pop()
                if sentence != start[1]:
                    print("Something mucked up", sentence, word, start)
                info[sentence, start[2], word +1] = start[0]
        word += 1
        if len(line) == 0:
            sentence += 1
            word = 0
    return info


def read_conll_speakers(lines):
    info = {}
    word = 0
    sentence = 0
    for line in lines:
        line = line.strip()
        fields = re.split(r'\s+', line)
        if len(fields) >= 10:
            spk_info = fields[9]
            if spk_info != '-' and len(spk_info) > 1:
                if sentence not in info:
                    info[sentence] = {}
                info[sentence][sentence, word, word + 1] = spk_info
        word += 1
        if len(line) == 0:
            sentence += 1
            word = 0
    return info

def read_conll_fcol(lines):
    info = [[]]
    for line in lines:
        line = line.strip()
        fields = re.split(r'\s+', line)
        if len(line) == 0:
            info.append([])
        else:
            info[-1].append(fields[0])
    if len(info[-1]) == 0:
        info.pop()
    return info


def read_conll_coref(lines):
    # Assumes:
    #  - Reading a single part
    #  - If duplicate mentions occur, use the first
    regex = "([(][0-9]*[)])|([(][0-9]*)|([0-9]*[)])|([|])"
    mentions = {} # (sentence, start, end+1) -> ID
    clusters = {} # ID -> list of (sentence, start, end+1)s
    unmatched_mentions = defaultdict(lambda: [])
    sentence = 0
    word = 0
    line_no = 0
    for line in lines:
        line_no += 1
        if len(line) > 0 and line[0] =='#':
            continue
        line = line.strip()
        if len(line) == 0:
            sentence += 1
            word = 0
            unmatched_mentions = defaultdict(lambda: [])
            continue
        # Canasai's comment out: fields = line.strip().split()
        fields = re.split(r'\s+', line.strip())
        for triple in re.findall(regex, fields[-1]):
            if triple[1] != '':
                val = int(triple[1][1:])
                unmatched_mentions[(sentence, val)].append(word)
            elif triple[0] != '' or triple[2] != '':
                start = word
                val = -1
                if triple[0] != '':
                    val = int(triple[0][1:-1])
                else:
                    val = int(triple[2][:-1])
                    if (sentence, val) not in unmatched_mentions:
                        print("Ignoring a mention with no start", str(val), line.strip(), line_no)
                        continue
                    if len(unmatched_mentions[(sentence, val)]) == 0:
                        print("No other start available", str(val), line.strip(), line_no)
                        continue
                    start = unmatched_mentions[(sentence, val)].pop()
                end = word + 1
                if (sentence, start, end) in mentions:
                    print("Duplicate mention", sentence, start, end, val, mentions[sentence, start, end])
                else:
                    mentions[sentence, start, end] = val
                    if val not in clusters:
                        clusters[val] = []
                    clusters[val].append((sentence, start, end))
        word += 1
    for key in unmatched_mentions:
        if len(unmatched_mentions[key]) > 0:
            print("Mention started, but did not end ", str(unmatched_mentions[key]))
    return mentions, clusters


def read_conll_doc(filename, ans=None, rtext=True, rparses=True, rheads=True, rclusters=True, rner=True, rspeakers=True, rfcol=False):
    if ans is None:
        ans = {}
    cur = []
    keys = None
    for line in codecs.open(filename, 'r', 'utf-8'):
        if len(line) > 0 and line.startswith('#begin') or line.startswith('#end'):

            if 'begin' in line:
                desc = line.split()
                location = desc[2].strip('();')
                keys = (location, desc[-1])

            if len(cur) > 0:
                if keys is None:
                    print("Error reading conll file - invalid #begin statemen\n", line)
                else:
                    info = {}
                    if rtext:
                        info['text'] = read_conll_text(cur)
                    if rparses:
                        # TODO: test
                        info['parses'] = read_conll_parses(cur)
                        if rheads:
                            info['heads'] = [head_finder.collins_find_heads(parse) for parse in info['parses']]
                    if rclusters:
                        info['mentions'], info['clusters'] = read_conll_coref(cur)
                    if rner:
                        info['ner'] = read_conll_ner(cur)
                    if rspeakers:
                        info['speakers'] = read_conll_speakers(cur)
                    if rfcol:
                        info['fcol'] = read_conll_fcol(cur)

                    if keys[0] not in ans:
                        ans[keys[0]] = {}
                    ans[keys[0]][keys[1]] = info
                    keys = None
            cur = []
        else:
            cur.append(line)
    return ans

def read_conll_raw_data(raw_data, ans=None, rtext=True, rparses=True, rheads=True, rclusters=True, rner=True, rspeakers=True, rfcol=False):
    processed = []
    for line in raw_data:
        if len(line.strip()) > 0:
            line = line +'\n'
        processed.append(line)

    if ans is None:
        ans = {}
    cur = []
    keys = None
    for line in processed:
        if len(line) > 0 and line.startswith('#begin') or line.startswith('#end'):

            if 'begin' in line:
                desc = line.split()
                location = desc[2].strip('();')
                keys = (location, desc[-1])

            if len(cur) > 0:
                if keys is None:
                    print("Error reading conll file - invalid #begin statemen\n", line)
                else:
                    info = {}
                    if rtext:
                        info['text'] = read_conll_text(cur)
                    if rparses:
                        # TODO: test
                        info['parses'] = read_conll_parses(cur)
                        if rheads:
                            info['heads'] = [head_finder.collins_find_heads(parse) for parse in info['parses']]
                    if rclusters:
                        info['mentions'], info['clusters'] = read_conll_coref(cur)
                    if rner:
                        info['ner'] = read_conll_ner(cur)
                    if rspeakers:
                        info['speakers'] = read_conll_speakers(cur)
                    if rfcol:
                        info['fcol'] = read_conll_fcol(cur)

                    if keys[0] not in ans:
                        ans[keys[0]] = {}
                    ans[keys[0]][keys[1]] = info
                    keys = None
            cur = []
        else:
            cur.append(line)
    return ans

