import scipy
import scipy.io
import random

from batching import *


def read_from_id(entity2ids):
    entity2id = {}
    id2entity = {}
    for line in entity2ids:
        if len(line) > 1:
            entity2id[line[0]] = int(line[1])
            id2entity[int(line[1])] = line[0]
    return entity2id, id2entity


def init_norm_Vector(relinit, entinit, embedding_size):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstent.append(tmp)
    assert embedding_size % len(lstent[0]) == 0
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)


def getID(train_set, valid_set, test_set):
    lstEnts = {}
    lstRels = {}
    for line in train_set:
        line = line.strip().split()
        if line[0] not in lstEnts:
            lstEnts[line[0]] = len(lstEnts)
        if line[2] not in lstEnts:
            lstEnts[line[2]] = len(lstEnts)
        if line[1] not in lstRels:
            lstRels[line[1]] = len(lstRels)

    for line in valid_set:
        line = line.strip().split()
        if line[0] not in lstEnts:
            lstEnts[line[0]] = len(lstEnts)
        if line[2] not in lstEnts:
            lstEnts[line[2]] = len(lstEnts)
        if line[1] not in lstRels:
            lstRels[line[1]] = len(lstRels)

    for line in test_set:
        line = line.strip().split()
        if line[0] not in lstEnts:
            lstEnts[line[0]] = len(lstEnts)
        if line[2] not in lstEnts:
            lstEnts[line[2]] = len(lstEnts)
        if line[1] not in lstRels:
            lstRels[line[1]] = len(lstRels)

    entity2id = []
    for entity in lstEnts:
        entity2id.append((entity, str(lstEnts[entity])))

    relation2id = []
    for relation in lstRels:
        relation2id.append((relation, str(lstRels[relation])))

    return entity2id, relation2id


def parse_line(line):
    line = line.strip().split()
    sub = line[0]
    rel = line[1]
    obj = line[2]
    val = [1]
    if len(line) > 3:
        if line[3] == '-1':
            val = [-1]
    return sub, obj, rel, val


def load_triples(dataset, words_indexes=None, parse_line=parse_line):
    """
    Take a list of file names and build the corresponding dictionnary of triples
    """
    if words_indexes == None:
        words_indexes = dict()
        entities = set()
        next_ent = 0
    else:
        entities = set(words_indexes)
        next_ent = max(words_indexes.values()) + 1

    data = dict()

    for _, line in enumerate(dataset):
        sub, obj, rel, val = parse_line(line)

        if sub in entities:
            sub_ind = words_indexes[sub]
        else:
            sub_ind = next_ent
            next_ent += 1
            words_indexes[sub] = sub_ind
            entities.add(sub)

        if rel in entities:
            rel_ind = words_indexes[rel]
        else:
            rel_ind = next_ent
            next_ent += 1
            words_indexes[rel] = rel_ind
            entities.add(rel)

        if obj in entities:
            obj_ind = words_indexes[obj]
        else:
            obj_ind = next_ent
            next_ent += 1
            words_indexes[obj] = obj_ind
            entities.add(obj)

        data[(sub_ind, rel_ind, obj_ind)] = val

    indexes_words = {}
    for tmpkey in words_indexes:
        indexes_words[words_indexes[tmpkey]] = tmpkey

    return data, words_indexes, indexes_words


def build_data(filename='./data/WN18', split_ratio=(0.7, 0.2, 0.1)):

    # split a file to train, valid, test
    processed_docs = []
    with open(filename, 'r') as f:
        for line in f:
            processed_docs.append(line.strip())

    train_dev_split_idx = int(len(processed_docs) * split_ratio[0])
    dev_test_split_idx = int(len(processed_docs) * (split_ratio[0] + split_ratio[1]))

    train_set, valid_set, test_set = processed_docs[ : train_dev_split_idx],\
                processed_docs[train_dev_split_idx : dev_test_split_idx],\
                processed_docs[dev_test_split_idx : ]


    train_triples, words_indexes, _ = load_triples(train_set, parse_line=parse_line)
    valid_triples, words_indexes, _ = load_triples(valid_set, words_indexes=words_indexes, parse_line=parse_line)
    test_triples, words_indexes, indexes_words = load_triples(test_set, words_indexes=words_indexes, parse_line=parse_line)


    entity2ids, relation2ids = getID(train_set, valid_set, test_set)  

    entity2id, id2entity = read_from_id(entity2ids)
    relation2id, id2relation = read_from_id(relation2ids)

    left_entity = {}
    right_entity = {}

    for _, line in enumerate(train_set):
        head, tail, rel, val = parse_line(line)
        # count the number of occurrences for each (heal, rel)
        if relation2id[rel] not in left_entity:
            left_entity[relation2id[rel]] = {}
        if entity2id[head] not in left_entity[relation2id[rel]]:
            left_entity[relation2id[rel]][entity2id[head]] = 0
        left_entity[relation2id[rel]][entity2id[head]] += 1
        # count the number of occurrences for each (rel, tail)
        if relation2id[rel] not in right_entity:
            right_entity[relation2id[rel]] = {}
        if entity2id[tail] not in right_entity[relation2id[rel]]:
            right_entity[relation2id[rel]][entity2id[tail]] = 0
        right_entity[relation2id[rel]][entity2id[tail]] += 1

    left_avg = {}
    for i in range(len(relation2id)):
        if i in left_entity:
            left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])
        else:
            left_avg[i] = 0

    right_avg = {}
    for i in range(len(relation2id)):
        if i in right_entity:
            right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])
        else:
            right_avg[i] = 0

    headTailSelector = {}
    for i in range(len(relation2id)):
        if (right_avg[i] + left_avg[i]) != 0:
            headTailSelector[i] = 1000 * right_avg[i] / (right_avg[i] + left_avg[i])
        else:
            headTailSelector[i] = 0

    return train_triples, valid_triples, test_triples, words_indexes, indexes_words, headTailSelector, entity2id, id2entity, relation2id, id2relation

def dic_of_chars(words_indexes):
    lstChars = {}
    for word in words_indexes:
        for char in word:
            if char not in lstChars:
                lstChars[char] = len(lstChars)
    lstChars['unk'] = len(lstChars)
    return lstChars


def convert_to_seq_chars(x_batch, lstChars, indexes_words):
    lst = []
    for [tmpH, tmpR, tmpT] in x_batch:
        wH = [lstChars[tmp] for tmp in indexes_words[tmpH]]
        wR = [lstChars[tmp] for tmp in indexes_words[tmpR]]
        wT = [lstChars[tmp] for tmp in indexes_words[tmpT]]
        lst.append([wH, wR, wT])
    return lst

def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    sequence_padded, sequence_length = [], []
    max_length_word = max([max(map(lambda x: len(x), seq))
                           for seq in sequences])
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return np.array(sequence_padded).astype(np.int32), np.array(sequence_length).astype(np.int32)
