# -*- coding: utf-8 -*-
import numpy as np
import configs
from keras.preprocessing.sequence import pad_sequences

def read_data(file_path):
    sentences = []
    tokens = []
    tags = []
    output_labels = []
    max_length = 0
    with open(file_path,'r',encoding='utf-8') as file:
        for line in file:
            if (line.split(' ')[0] == "-DOCSTART-"):
                continue
            if (line == '\n'):
                if(len(tokens)>0 or (max_length == 0)):
                    if(len(tokens) > max_length ):
                       max_length = len(tokens)
                    tokens = []
                    tags = []
                    sentences.append(tokens)
                    output_labels.append(tags)
                else:
                    continue
            else:
                line_data = line.strip().split(' ')
                tokens.append(line_data[0])
                if len(line_data) > 4:
                    tags.append(line_data[3])
                else:
                    tags.append(line_data[0])

        sentences.pop(-1)
        output_labels.pop(-1)
    return sentences, output_labels, max_length       
                 

def get_vocabulory(sentences):
    vocabs = {'PAD': 0}
    i = 1
    for tokens in sentences:
        for word in tokens:
            if word not in vocabs:
                vocabs[word] = i
                i+=1               
    return vocabs 


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding = 'utf-8') as file:
        word_to_vec_map = {}
        for line in file:
            line = line.strip().split()
            curr_word = line[0]
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map     
    

def get_preTrained_embeddings(word_to_index,glove_vectors,vocab_size):
    embed_dim = configs.EMBEDDING_DIM
    embed_matrix = np.zeros((vocab_size+1, embed_dim))
    add_words = []
    for word,i in word_to_index.items():
        if i <= 0:
            continue
        if word in glove_vectors:
            embed_vector =  glove_vectors[word.lower()]
            embed_matrix[i] = embed_vector
        elif word.lower() in glove_vectors:
            embed_vector =  glove_vectors[word.lower()]
            embed_matrix[i] = embed_vector
            add_words.append(word)      
        else:
            embed_matrix[i] = np.random.normal(embed_dim)       
#    for word in add_words:
#        word_to_index[word.lower()] = word_to_index[word]
    return embed_matrix          
    
def prepare_outputs(output_labels):
    unq_labels = {}
    for labels in output_labels:
        for label in labels:
             if label not in unq_labels:
                unq_labels[label] = 1
             else:
                unq_labels[label] += 1
    unq_labels = list(sorted(unq_labels, key=unq_labels.__getitem__, reverse=True))
    i=0           
    label_to_index = {}
    index_to_label = {}
    for label in unq_labels:
        label_to_index[label] = i
        index_to_label[i] = label
        i += 1
    return label_to_index, index_to_label    

def get_sequence_indices(sentences, word_to_index, max_length):
      no_of_examples  = len(sentences)
      sequences  = np.zeros((no_of_examples, max_length), dtype = np.int32)
      for i, sentence in enumerate(sentences):
          for j, word in enumerate(sentence):
              if word in word_to_index:
                  sequences[i,j] =  word_to_index[word]
              elif word.lower() in word_to_index:
                  sequences[i,j] =  word_to_index[word.lower()]
                      
      return sequences


def get_orig_labels(indices, index_to_label,ref_labels):
    seq_labels = []
    labels = []
    for i,index in enumerate(indices):
        for jval in range(len(ref_labels[i])):
            labels.append(index_to_label[index[jval]])
        seq_labels.append(labels)
        labels = []
    return seq_labels

def get_chars(sentences, max_length, char_index):
    max_wordlength= configs.MAX_CHARS
    char_seqs  = []
    chars = []
    for i, sentence in enumerate(sentences):
        for j,word in enumerate(sentence):
            if len(word)<=max_wordlength:
                chars.append([char_index[c] for c in word if c in char_index])
        chars = pad_sequences(chars,max_wordlength,padding='post')
        char_seqs.append(chars)
        chars = []
    char_seqs = pad_sequences(char_seqs,max_length,padding='post')
    return np.asarray(char_seqs)