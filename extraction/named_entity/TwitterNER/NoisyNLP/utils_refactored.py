import os, glob, itertools
from collections import defaultdict

def conll2012_to_ditk(lines):
    converted_lines =[]

    flag = None
    for line in lines:
        #print line
        l = line.strip()
        #print (l)
        #' '.join((agent_contact, agent_telno)).encode('utf-8').strip()
        #if len(l) > 0 and ' '.join(l).startswith('#begin').encode('utf-8') or ' '.join(l).startswith('#begin').encode('utf-8'):
           # continue
        if len(l) > 0 and l.encode('utf-8').startswith('#begin') or l.encode('utf-8').startswith('#begin'):
            continue
        l = ' '.join(l.split())
        ls = l.split(" ")
        #print ls
        converted_line = []
        #print (ls)

        if len(ls) >= 11:
            extra = ls[0:3]  # 4-7
            extra.extend(ls[6:10])  # 7 - 11
            extra.extend(ls[11:])  # 11:
            word = ls[3]  # 0
            #print (word)
            pos = ls[4]  # 1
            cons = ls[5]  # 2
            ori_ner = ls[10]  # 3
            ner = ori_ner
            # print(word, pos, cons, ner)

            if ori_ner == "*":
                if flag == None:
                    ner = "O"
                else:
                    ner = "I-" + flag
            elif ori_ner == "*)":
                ner = "I-" + flag
                flag = None
            elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner) > 2:
                flag = ori_ner[1:-1]
                ner = "B-" + flag
            elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner) > 2 and flag == None:
                ner = "B-" + ori_ner[1:-1]

            converted_line.extend([word, pos, cons, ner])
            #print (converted_line)
            #converted_line.extend(extra)

            # text += " ".join([word, pos, cons, ner]) + " "
            # text += " ".join(extra) + '\n'
            converted_lines.append(converted_line)
            # print(j)
        else:
            # text += '\n'
            converted_lines.append(line)
    # text += '\n'

    return converted_lines

def load(filename, sep="\t", notypes=False):
    tag_count = defaultdict(int)
    sequences = []
    with open(filename) as fp:
        seq = []
        for line in fp:
          #print seq
          #print sequences
          line = line.strip()
          if line:
            line = line.split(sep)
            if notypes:
              line[1] = line[1][0]
            try:
              tag_count[line[1]] += 1
              #print line
              seq.append(Tag(*line))
            except:
              pass
            else:
              sequences.append(seq)
              seq = []

        if seq:
            sequences.append(seq)
    return sequences, tag_count

def write_sequences(sequences, filename, sep="\t", to_bieou=True):
    with open(filename, "wb+") as fp:
        for seq in sequences:
            if to_bieou:
                seq = to_BIEOU(seq)
            for tag in seq:
                print >> fp, sep.join(tag).encode('utf-8')
            print >> fp, ""

def write_sequences(sequences, filename, sep="\t", to_bieou=True):
    with open(filename, "wb+") as fp:
        for seq in sequences:
            if to_bieou:
                seq = to_BIEOU(seq)
            for tag in seq:
                print >> fp, sep.join(tag).encode('utf-8')
            print >> fp, ""

def phrase_to_BIEOU(phrase):
    l = len(phrase)
    new_phrase = []
    for j, t in enumerate(phrase):
        new_tag = t.tag
        if l == 1:
            new_tag = "U%s" % t.tag[1:]
        elif j == l-1:
            new_tag = "E%s" % t.tag[1:]
        new_phrase.append(Tag(t.token, new_tag))
    return new_phrase

def to_BIEOU(seq, verbose=False):
    # TAGS B I E U O
    phrase = []
    new_seq = []
    for i, tag in enumerate(seq):
        if not phrase and tag.tag[0] == "B":
            phrase.append(tag)
            continue
        if tag.tag[0] == "I":
            phrase.append(tag)
            continue
        if phrase:
            if verbose:
                print "Editing phrase", phrase
            new_phrase = phrase_to_BIEOU(phrase)
            new_seq.extend(new_phrase)
            phrase = []
        new_seq.append(tag)
    if phrase:
        if verbose:
            print "Editing phrase", phrase
            new_phrase = phrase_to_BIEOU(phrase)
            new_seq.extend(new_phrase)
            phrase = []
        new_seq.append(tag)
    if phrase:
        if verbose:
            print "Editing phrase", phrase
        new_phrase = phrase_to_BIEOU(phrase)
        new_seq.extend(new_phrase)
        phrase = []
    return new_seq

def write_to_csv(input,filePath):
    fp=open(filePath,'w+')
    for each_input in input:
        text=""
        try:
            text+=each_input[0]+'\t'+each_input[1]+'\t'+each_input[2]+'\t'+each_input[3]+'\n'
            fp.write(text)
        except:
            pass

    fp.close()
