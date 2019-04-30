import re
import os
import sys
import codecs
import subprocess
from collections import deque, defaultdict

sys.path.append("/home/danniel/Desktop/CONLL2012-intern")
import load_conll_2012.pstree

from rnn import Node

def get_reversed_head_list(head_list):
    words = len(head_list)
    left_list = [[] for i in head_list]
    right_list = [[] for i in head_list]
    root = -1
    
    for index, head in enumerate(head_list):
        if head == -1:
            root = index
        elif index < head:
            left_list[head].append(index)
        elif index > head:
            right_list[head].append(index)
    
    reversed_head_list = []
    for index in range(len(head_list)):
        reversed_head_list.append(left_list[index][::-1] + right_list[index])
    return reversed_head_list, root
    
def read_conllu(conllu_path):
    with open(conllu_path, "r") as f:
        line_list = f.read().splitlines()
    
    sentence_list = []
    
    word_list, pos_list, head_list, relation_list = [], [], [], []
    for line in line_list:
        if not line:
            head_list, root = get_reversed_head_list(head_list)
            sentence_list.append([word_list, pos_list, head_list, relation_list, root])
            word_list, pos_list, head_list, relation_list = [], [], [], []
            continue
        _, word, _, _, pos, _, head, relation, _, _ = line.split("\t")
        word_list.append(word)
        pos_list.append(pos)
        head_list.append(int(head)-1)
        relation_list.append(relation)
    
    return sentence_list

def dependency_to_constituency(word_list, pos_list, head_list, relation_list, index):
    leaf = Node()
    leaf.word = word_list[index]
    leaf.pos = pos_list[index]
    leaf.span = (index, index+1)
    leaf.head = leaf.word
    
    root = leaf
    for child_index in head_list[index]:
        child_root = dependency_to_constituency(word_list, pos_list, head_list, relation_list,
                            child_index)
        new_root = Node()
        new_root.word = None
        new_root.pos = relation_list[child_index]
        if child_index < index:
            new_root.span = (child_root.span[0], root.span[1])
            new_root.add_child(child_root)
            new_root.add_child(root)
        else:
            new_root.span = (root.span[0], child_root.span[1])
            new_root.add_child(root)
            new_root.add_child(child_root)
        new_root.head = root.head
        root = new_root
    
    return root
    
def show_tree(node, indent):
    word = node.word if node.word else ""
    print(indent + node.pos + " " + word + " " + repr(node.span))
    
    for child in node.child_list:
        show_tree(child, indent+"    ")
    return
    
if __name__ == "__main__":
    #read_conllu("conll2003dep/dependency_train.conllu")
    
    """
    rlist, root = get_reversed_head_list([3,3,3,-1,6,6,3,6,11,10,11,6])
    print "root", root
    for i, j in enumerate(rlist):
        print i, j
    """
    
    sentence_list = read_conllu("conll2003dep/tmp.conllu")
    sentence = sentence_list[0]
    root = dependency_to_constituency(*sentence)
    show_tree(root, "")
    
    exit()