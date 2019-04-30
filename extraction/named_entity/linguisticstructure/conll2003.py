import re
import os
import sys
import codecs
import subprocess
from collections import defaultdict

import numpy as np

# sys.path.append("/home/danniel/Desktop/CONLL2012-intern")
import load_conll_2012.pstree as pstree
import load_conll_2012.head_finder as head_finder

from rnn import Node

kk_base_path = "D:/Study/CSCI 548 IIW/Project/Resources/TF_RNN/"
data_path = kk_base_path + "conll-2003"
dataset = "conll2003"

glove_file = kk_base_path + "glove.840B.300d/glove.840B.300d.txt"
senna_path = kk_base_path + "/senna/hash"
parser_classpath = kk_base_path + "stanford-parser-full-2015-12-09/*;"


character_file = os.path.join(dataset, "character.txt")
word_file = os.path.join(dataset, "word.txt")
pos_file = os.path.join(dataset, "pos.txt")
ne_file = os.path.join(dataset, "ne.txt")
pretrained_word_file = os.path.join(dataset, "word.npy")
pretrained_embedding_file = os.path.join(dataset, "embedding.npy")

lexicon_meta_list = [
    {"ne": "PER",  "path": os.path.join(dataset, "senna_per.txt"),  "senna": os.path.join(senna_path, "ner.per.lst")},
    {"ne": "ORG",  "path": os.path.join(dataset, "senna_org.txt"),  "senna": os.path.join(senna_path, "ner.org.lst")},
    {"ne": "LOC",  "path": os.path.join(dataset, "senna_loc.txt"),  "senna": os.path.join(senna_path, "ner.loc.lst")},
    {"ne": "MISC", "path": os.path.join(dataset, "senna_misc.txt"), "senna": os.path.join(senna_path, "ner.misc.lst")}
    ]


split_raw = {"train": "eng.train", "dev": "eng.testa", "test": "eng.testb"}

split_sentence = {"train": "sentence_train.txt", "dev": "sentence_dev.txt", "test": "sentence_test.txt"}
split_parse = {"train": "parse_train.txt", "dev": "parse_dev.txt", "test": "parse_test.txt"}

def log(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    return

def read_list_file(file_path, encoding="utf8"):
    log("Read %s..." % file_path)

    with codecs.open(file_path, "r", encoding=encoding) as f:
        line_list = f.read().splitlines()
    line_to_index = {line: index for index, line in enumerate(line_list)}

    log(" %d lines\n" % len(line_to_index))
    return line_list, line_to_index

def group_sequential_label(seq_ne_list):
    span_ne_dict = {}

    start, ne = -1, None
    for index, label in enumerate(seq_ne_list + ["O"]):
        if (label[0]=="O" or label[0]=="B") and ne:
            span_ne_dict[(start, index)] = ne
            start, ne = -1, None

        if label[0]=="B" or (label[0]=="I" and not ne):
            start, ne = index, label[2:]

    return span_ne_dict

def extract_ner(split):
    #with open("tmp.txt", "r") as f:
    with open(os.path.join(data_path, split_raw[split]), "r") as f:
        line_list = f.read().splitlines()

    sentence_list = []
    ner_list = []

    sentence = []
    ner = []
    for line in line_list[2:]:
        if line[:10] == "-DOCSTART-": continue
        if not line:
            if sentence:
                sentence_list.append(sentence)
                ner_list.append(group_sequential_label(ner))
                sentence = []
                ner = []
            continue
        word, _, _, sequential_label = line.split()
        sentence.append(word)
        ner.append(sequential_label)
    """
    for i, j in enumerate(sentence_list):
        print(""
        print(j
        print(ner_list[i]
    """
    return sentence_list, ner_list

def extract_pos_from_pstree(tree, pos_set):
    pos_set.add(tree.label)

    for child in tree.subtrees:
        extract_pos_from_pstree(child, pos_set)
    return

def print_pstree(node, indent):
    word = node.word if node.word else ""
    print(indent + node.label + " "+ word)

    for child in node.subtrees:
        print_pstree(child, indent+"    ")
    return

def is_dataset_prepared():
    # TODO if exists file for parses, continue
    return True

def is_embedding_presaved():
    #TODO if extracted embeddings exist, continue.
    return True

def prepare_dataset():
    ne_set = set()
    word_set = set()
    character_set = set()
    pos_set = set()

    for split in split_raw:
        sentence_list, ner_list = extract_ner(split)

        # Procecss raw NER
        for ner in ner_list:
            for ne in ner.values():
                ne_set.add(ne)

        # Procecss raw sentences
        split_sentence_file = os.path.join(dataset, split_sentence[split])
        with open(split_sentence_file, "w") as f:
            for sentence in sentence_list:
                f.write(" ".join(sentence)+"\n")
                for word in sentence:
                    word_set.add(word)
                    for character in word:
                        character_set.add(character)
        word_set |= {"``", "''", "-LSB-", "-RSB-", "-LRB-", "25.49,-LRB-3-yr", "6-7-LRB-3-7", "Videoton-LRB-*", "-RRB-",
                     "1-RRB-266", "12.177-RRB-.", "53.04-RRB-.", "Austria-RRB-118"}

        split_parse_file = os.path.join(dataset, split_parse[split])

        # Generate parses
        with open(split_parse_file, "w") as f:
            subprocess.call(["java", "-cp", parser_classpath,
                "edu.stanford.nlp.parser.lexparser.LexicalizedParser",
                "-outputFormat", "oneline", "-sentences", "newline", "-tokenized",
                "-escaper", "edu.stanford.nlp.process.PTBEscapingProcessor",
                "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz",
                split_sentence_file], stdout=f)

        # Process parses
        with open(split_parse_file, "r") as f:
            line_list = f.read().splitlines()
        for line in line_list:
            tree = pstree.tree_from_text(line)
            extract_pos_from_pstree(tree, pos_set)

    with open(ne_file, "w") as f:
        for ne in sorted(ne_set):
            f.write(ne + '\n')

    with open(word_file, "w") as f:
        for word in sorted(word_set):
            f.write(word + '\n')

    with open(character_file, "w") as f:
        for character in sorted(character_set):
            f.write(character + '\n')

    with open(pos_file, "w") as f:
        for pos in sorted(pos_set):
            f.write(pos + '\n')
    return

def extract_glove_embeddings():
    log("extract_glove_embeddings()...")

    _, word_to_index = read_list_file(word_file)
    word_list = []
    embedding_list = []
    with open(glove_file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if len(line) > 301:
                idx = len(line) - 300
                temp = line[0: idx]
                temp = ' '.join(temp)
                npa = line[idx:]
            else:
                temp = line[0]
                npa = line[1:]

            word = temp
            if word not in word_to_index: continue
            try:
                embedding = np.array([float(i) for i in npa])
            except Exception:
                print(line)
            word_list.append(word)
            embedding_list.append(embedding)

    np.save(pretrained_word_file, word_list)
    np.save(pretrained_embedding_file, embedding_list)

    log(" %d pre-trained words\n" % len(word_list))
    return

def construct_node(node, tree, ner_raw_data, head_raw_data, text_raw_data,
        character_to_index, word_to_index, pos_to_index, index_to_lexicon,
        pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node):
    pos = tree.label
    word = tree.word
    span = tree.span
    head = tree.head if hasattr(tree, "head") else head_raw_data[(span, pos)][1]
    ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
    constituent = " ".join(text_raw_data[span[0]:span[1]]).lower()

    # Process pos info
    node.pos_index = pos_to_index[pos]
    pos_count[pos] += 1

    # Process word info
    node.word_split = [character_to_index[character] for character in word] if word else []
    node.word_index = word_to_index[word] if word else -1

    # Process head info
    node.head_split = [character_to_index[character] for character in head]
    #if head == "-LSB-": print(text_raw_data
    node.head_index = word_to_index[head]

    # Process ne info
    node.ne = ne
    if ne != "NONE":
        if not node.parent or node.parent.span!=span:
            ne_count[ne] += 1
        pos_ne_count[pos] += 1

    # Process span info
    node.span = span
    span_to_node[span] = node

    # Process lexicon info
    node.lexicon_hit = [0] * len(index_to_lexicon)
    hits = 0
    for index, lexicon in index_to_lexicon.items():
        if constituent in lexicon:
            node.lexicon_hit[index] = 1
            hits = 1
    lexicon_hits[0] += hits

    # Binarize children
    if len(tree.subtrees) > 2:
        side_child_pos = tree.subtrees[-1].label
        side_child_span = tree.subtrees[-1].span
        side_child_head = head_raw_data[(side_child_span, side_child_pos)][1]
        if side_child_head != head:
            sub_subtrees = tree.subtrees[:-1]
        else:
            sub_subtrees = tree.subtrees[1:]
        new_span = (sub_subtrees[0].span[0], sub_subtrees[-1].span[1])
        new_tree = pstree.PSTree(label=pos, span=new_span, subtrees=sub_subtrees)
        new_tree.head = head
        if side_child_head != head:
            tree.subtrees = [new_tree, tree.subtrees[-1]]
        else:
            tree.subtrees = [tree.subtrees[0], new_tree]

    # Process children
    nodes = 1
    for subtree in tree.subtrees:
        child = Node()
        node.add_child(child)
        child_nodes = construct_node(child, subtree, ner_raw_data, head_raw_data, text_raw_data,
            character_to_index, word_to_index, pos_to_index, index_to_lexicon,
            pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node)
        nodes += child_nodes
    return nodes

def create_dense_nodes(ner_raw_data, text_raw_data, pos_to_index, index_to_lexicon,
        pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node):
    node_list = []
    max_dense_span = 3
    # Start from bigram, since all unigrams are already covered by parses
    for span_length in range(2, 1+max_dense_span):
        for span_start in range(0, 1+len(text_raw_data)-span_length):
            span = (span_start, span_start+span_length)
            if span in span_to_node: continue
            pos = "NONE"
            ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
            constituent = " ".join(text_raw_data[span[0]:span[1]]).lower()

            # span, child
            # TODO: sibling
            node = Node()
            node_list.append(node)
            node.span = span
            span_to_node[span] = node
            node.child_list = [span_to_node[(span[0],span[1]-1)], span_to_node[(span[0]+1,span[1])]]

            # word, head, pos
            node.pos_index = pos_to_index[pos]
            pos_count[pos] += 1
            node.word_split = []
            node.word_index = -1
            node.head_split = []
            node.head_index = -1

            # ne
            node.ne = ne
            if ne != "NONE":
                ne_count[ne] += 1
                pos_ne_count[pos] += 1

            # lexicon
            node.lexicon_hit = [0] * len(index_to_lexicon)
            hits = 0
            for index, lexicon in index_to_lexicon.items():
                if constituent in lexicon:
                    node.lexicon_hit[index] = 1
                    hits = 1
            lexicon_hits[0] += hits

    return node_list

def get_tree_data(sentence_list, parse_list, ner_list,
        character_to_index, word_to_index, pos_to_index, index_to_lexicon):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL-2003
    
    Stores into Node data structure
    """
    tree_pyramid_list = []
    word_count = 0
    pos_count = defaultdict(lambda: 0)
    ne_count = defaultdict(lambda: 0)
    pos_ne_count = defaultdict(lambda: 0)
    lexicon_hits = [0]

    for index, parse in enumerate(parse_list):
        text_raw_data = sentence_list[index]
        word_count += len(text_raw_data)
        span_to_node = {}
        head_raw_data = head_finder.collins_find_heads(parse)

        root_node = Node()
        nodes = construct_node(
           root_node, parse, ner_list[index], head_raw_data, text_raw_data,
           character_to_index, word_to_index, pos_to_index, index_to_lexicon,
           pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node)
        root_node.nodes = nodes
        root_node.tokens = len(text_raw_data)

        additional_node_list = create_dense_nodes(
            ner_list[index], text_raw_data,
            pos_to_index, index_to_lexicon,
            pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node)

        tree_pyramid_list.append((root_node, additional_node_list))

    log(" %d sentences\n" % len(tree_pyramid_list))
    return tree_pyramid_list, word_count, pos_count, ne_count, pos_ne_count, lexicon_hits[0]

def label_tree_data(node, pos_to_index, ne_to_index):
    node.y = ne_to_index[node.ne]
    # node.y = ne_to_index[":".join(node.ner)]

    for child in node.child_list:
        label_tree_data(child, pos_to_index, ne_to_index)
    return

def read_dataset(data_split_list = ["train", "dev", "test"]):
    # Read all raw data
    sentence_data = {}
    ner_data = {}
    parse_data = {}
    for split in data_split_list:
        sentence_data[split], ner_data[split] = extract_ner(split)

        split_parse_file = os.path.join(dataset, split_parse[split])
        with open(split_parse_file, "r") as f:
            line_list = f.read().splitlines()
        parse_data[split] = [pstree.tree_from_text(line) for line in line_list]

    # Read lists of annotations
    character_list, character_to_index = read_list_file(character_file)
    word_list, word_to_index = read_list_file(word_file)
    pos_list, pos_to_index = read_list_file(pos_file)
    ne_list, ne_to_index = read_list_file(ne_file)

    pos_to_index["NONE"] = len(pos_to_index)

    # Read lexicon
    index_to_lexicon = {}
    for index, meta in enumerate(lexicon_meta_list):
        _, index_to_lexicon[index] = read_list_file(meta["senna"], "iso8859-15")

    # Build a tree structure for each sentence
    data = {}
    word_count = {}
    pos_count = {}
    ne_count = {}
    pos_ne_count = {}
    lexicon_hits = {}

    for split in data_split_list:
        (tree_pyramid_list,
            word_count[split], pos_count[split], ne_count[split], pos_ne_count[split],
            lexicon_hits[split]) = get_tree_data(
                sentence_data[split], parse_data[split], ner_data[split],
                character_to_index, word_to_index, pos_to_index, index_to_lexicon)
        data[split] = {"tree_pyramid_list": tree_pyramid_list, "ner_list": ner_data[split]}

    # Compute the mapping to labels
    ne_to_index["NONE"] = len(ne_to_index)

    # Add label to nodes
    for split in data_split_list:
        for tree, pyramid in data[split]["tree_pyramid_list"]:
            label_tree_data(tree, pos_to_index, ne_to_index)
            for node in pyramid:
                node.y = ne_to_index[node.ne]

    return (data, word_list, ne_list,
        len(character_to_index), len(pos_to_index), len(ne_to_index), len(index_to_lexicon))

def extract_ner_from_raw(data):
    line_list = data
    sentence_list = []
    ner_list = []

    sentence = []
    ner = []
    for line in line_list[2:]:
        if line[:10] == "-DOCSTART-": continue
        if not line:
            if sentence:
                sentence_list.append(sentence)
                ner_list.append(group_sequential_label(ner))
                sentence = []
                ner = []
            continue
        word, _, _, sequential_label = line.split()
        sentence.append(word)
        ner.append(sequential_label)
    """    
    for i, j in enumerate(sentence_list):
        print(""
        print(j
        print(ner_list[i]
    """
    return sentence_list, ner_list

def prepare_raw_data(data):
    ne_set = set()
    word_set = set()
    character_set = set()
    pos_set = set()

    for split in split_raw:
        sentence_list, ner_list = extract_ner_from_raw(data[split])

        # Procecss raw NER
        for ner in ner_list:
            for ne in ner.values():
                ne_set.add(ne)

        # Procecss raw sentences
        split_sentence_file = os.path.join(dataset, split_sentence[split])
        with open(split_sentence_file, "w") as f:
            for sentence in sentence_list:
                f.write(" ".join(sentence)+"\n")
                for word in sentence:
                    word_set.add(word)
                    for character in word:
                        character_set.add(character)
        word_set |= {"``", "''", "-LSB-", "-RSB-", "-LRB-", "25.49,-LRB-3-yr", "6-7-LRB-3-7", "Videoton-LRB-*", "-RRB-",
                     "1-RRB-266", "12.177-RRB-.", "53.04-RRB-.", "Austria-RRB-118"}

        split_parse_file = os.path.join(dataset, split_parse[split])

        # Generate parses
        with open(split_parse_file, "w") as f:
            subprocess.call(["java", "-cp", parser_classpath,
                "edu.stanford.nlp.parser.lexparser.LexicalizedParser",
                "-outputFormat", "oneline", "-sentences", "newline", "-tokenized",
                "-escaper", "edu.stanford.nlp.process.PTBEscapingProcessor",
                "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz",
                split_sentence_file], stdout=f)

        # Process parses
        with open(split_parse_file, "r") as f:
            line_list = f.read().splitlines()
        for line in line_list:
            tree = pstree.tree_from_text(line)
            extract_pos_from_pstree(tree, pos_set)

    with open(ne_file, "w") as f:
        for ne in sorted(ne_set):
            f.write(ne + '\n')

    with open(word_file, "w") as f:
        for word in sorted(word_set):
            f.write(word + '\n')

    with open(character_file, "w") as f:
        for character in sorted(character_set):
            f.write(character + '\n')

    with open(pos_file, "w") as f:
        for pos in sorted(pos_set):
            f.write(pos + '\n')
    return

def preprocess_raw_dataset(data, data_split_list=["train", "dev", "test"], ne_tags=None, pos_tags=None, senna_hash_path=None, glove_file_path=None):

    # Read all raw data
    prepare_raw_data(data)
    # extract_glove_embeddings()
    sentence_data = {}
    ner_data = {}
    parse_data = {}
    for split in data_split_list:
        sentence_data[split], ner_data[split] = extract_ner_from_raw(data[split])

        split_parse_file = os.path.join(dataset, split_parse[split])
        with open(split_parse_file, "r") as f:
            line_list = f.read().splitlines()
        parse_data[split] = [pstree.tree_from_text(line) for line in line_list]

    # Read lists of annotations
    character_list, character_to_index = read_list_file(character_file)
    word_list, word_to_index = read_list_file(word_file)
    pos_list, pos_to_index = read_list_file(pos_file)
    ne_list, ne_to_index = read_list_file(ne_file)

    pos_to_index["NONE"] = len(pos_to_index)
    global glove_file
    global senna_path

    if glove_file_path: glove_file = glove_file_path
    if senna_hash_path: senna_path = senna_hash_path

    # Read lexicon
    index_to_lexicon = {}
    for index, meta in enumerate(lexicon_meta_list):
        _, index_to_lexicon[index] = read_list_file(meta["senna"], "iso8859-15")

    # Build a tree structure for each sentence
    data = {}
    word_count = {}
    pos_count = {}
    ne_count = {}
    pos_ne_count = {}
    lexicon_hits = {}

    for split in data_split_list:
        (tree_pyramid_list,
            word_count[split], pos_count[split], ne_count[split], pos_ne_count[split],
            lexicon_hits[split]) = get_tree_data(
                sentence_data[split], parse_data[split], ner_data[split],
                character_to_index, word_to_index, pos_to_index, index_to_lexicon)
        data[split] = {"tree_pyramid_list": tree_pyramid_list, "ner_list": ner_data[split]}

    # Compute the mapping to labels
    ne_to_index["NONE"] = len(ne_to_index)

    # Add label to nodes
    for split in data_split_list:
        for tree, pyramid in data[split]["tree_pyramid_list"]:
            label_tree_data(tree, pos_to_index, ne_to_index)
            for node in pyramid:
                node.y = ne_to_index[node.ne]

    return (data, word_list, ne_list,
        len(character_to_index), len(pos_to_index), len(ne_to_index), len(index_to_lexicon))


def preprocess_data(raw_data, split='test'):
    sentence_data, ner_data = extract_ner_from_raw(raw_data)

    split_parse_file = os.path.join(dataset, split_parse[split])
    with open(split_parse_file, "r") as f:
        line_list = f.read().splitlines()
    parse_data = [pstree.tree_from_text(line) for line in line_list]

    character_list, character_to_index = read_list_file(character_file)
    word_list, word_to_index = read_list_file(word_file)
    pos_list, pos_to_index = read_list_file(pos_file)
    ne_list, ne_to_index = read_list_file(ne_file)

    pos_to_index["NONE"] = len(pos_to_index)
    index_to_lexicon = {}
    for index, meta in enumerate(lexicon_meta_list):
        _, index_to_lexicon[index] = read_list_file(meta["senna"], "iso8859-15")

    (tree_pyramid_list,
     word_count, pos_count, ne_count, pos_ne_count,
     lexicon_hits) = get_tree_data(
        sentence_data, parse_data, ner_data,
        character_to_index, word_to_index, pos_to_index, index_to_lexicon)
    data = {"tree_pyramid_list": tree_pyramid_list, "ner_list": ner_data}

    # Compute the mapping to labels


    ne_to_index["NONE"] = len(ne_to_index)

    # Add label to nodes
    for tree, pyramid in data["tree_pyramid_list"]:
        label_tree_data(tree, pos_to_index, ne_to_index)
        for node in pyramid:
            node.y = ne_to_index[node.ne]

    return data


if __name__ == "__main__":
    """
    print(""
    parse_string = "(ROOT (S (NP (NNP EU)) (VP (VBZ rejects) (NP (JJ German) (NN call)) (PP (TO to) (NP (NN boycott) (JJ British) (NN lamb)))) (. .)))"
    root = pstree.tree_from_text(parse_string)
    print_pstree(root, "")
    print(""
    for i, j in head_finder.collins_find_heads(root).items(): print(i, j
    """
    # prepare_dataset()
    # extract_glove_embeddings()
    # read_dataset()
    # read_data()
    exit()











