
import nltk
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import os
import re
from utils import readConfig
java_path = "C:/Program Files/Java/jdk1.8.0_11/bin/java.exe"
os.environ['JAVAHOME'] = java_path
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')

class Process_Data(object):
    def __init__(self):
        pass

    def preprocess_data(self, datasetName, fileName, configfile):

        config = readConfig(configfile)

        train_id2sent = dict()
        train_id2pos = dict()
        train_id2ner = dict()
        train_id2nerBILOU = dict()
        train_id2arg2rel = dict()

        seed_id=1000
        file = open(fileName, 'r')
        st = StanfordNERTagger(config['basedir']+config['ner_classfile'], config['basedir']+config['ner_jar'],encoding='utf-8')

        for line in file:
            line_arr = line.split('\t')
            sentence = line_arr[0]
            sentence = re.sub(r"[^a-zA-Z0-9]+", ' ', sentence)
            sentence = sentence.strip()
            entity1 = line_arr[1]
            entity1 = re.sub(r"[^a-zA-Z0-9]+", ' ', entity1)
            entity1 = entity1.strip()
            entity2 = line_arr[5]
            entity2 = re.sub(r"[^a-zA-Z0-9]+", ' ', entity2)
            entity2 = entity2.strip()
            relation = line_arr[9].replace('\n','').replace('\r','')

            entity1_pos=[]
            for str in entity1.split(" "):
                entity1_pos.append(sentence.split(" ").index(str))
            entity2_pos = []
            for str in entity2.split(" "):
                entity2_pos.append(sentence.split(" ").index(str))


            relation_set = set()
            for index1 in entity1_pos:
                for index2 in entity2_pos:
                    relation_set.add((index1,index2))

            relation_dict=dict()
            for pair in relation_set:
                relation_dict[pair] = relation

            token_text = word_tokenize(sentence)
            ne_tagged = st.tag(token_text)
            ne_tag_token1 = st.tag([entity1])
            ne_tag_token2 = st.tag([entity2])

            bio_tagged = []
            ner_list = []

            prev_tag = "O"
            for i in range(len(ne_tagged)):
                token = ne_tagged[i][0]
                tag = ne_tagged[i][1]

                if i >= entity1_pos[0] and i <= entity1_pos[len(entity1_pos)-1]:
                    tag = ne_tag_token1[0][1]
                    if tag == 'O':
                        tag='Other'

                if i >= entity2_pos[0] and i <= entity2_pos[len(entity2_pos)-1]:
                    tag = ne_tag_token2[0][1]
                    if tag == 'O':
                        tag='Other'

                if tag == "PERSON":
                    tag = "Peop"
                elif tag == "ORGANIZATION":
                    tag = "Org"
                elif tag == "LOCATION":
                    tag = "Loc"
                elif tag == "O":
                    tag = "O"
                else:
                    tag = "Other"
                ner_list.append(tag)
                if tag == "O":  # O
                    bio_tagged.append((token, tag))
                    prev_tag = tag
                    continue
                if tag != "O" and prev_tag == "O":  # Begin NE
                    bio_tagged.append((token, "B-" + tag))
                    prev_tag = tag
                elif prev_tag != "O" and prev_tag == tag:  # Inside NE
                    bio_tagged.append((token, "I-" + tag))
                    prev_tag = tag
                elif prev_tag != "O" and prev_tag != tag:  # Adjacent NE
                    bio_tagged.append((token, "B-" + tag))
                    prev_tag = tag


            tokens_list, ne_tags = zip(*bio_tagged)
            pos_list = [pos for token, pos in pos_tag(tokens_list)]

            bilou_list=[]
            prev_iob = "O"
            curr_iob = "O"
            for i in range(len(ne_tags)):
                curr_iob = ne_tags[i][0]
                if i < len(ne_tags)-1:
                    next_iob = ne_tags[i+1][0]
                if curr_iob == "O":
                    bilou_list.append(curr_iob)
                elif prev_iob == "O" and next_iob == "O":
                    bilou_list.append("U-"+ne_tags[i][2:])
                elif prev_iob == "O" and next_iob != "O":
                    bilou_list.append("B-" + ne_tags[i][2:])
                elif prev_iob != "O" and next_iob == "O":
                    bilou_list.append("L-" + ne_tags[i][2:])
                elif prev_iob != "O" and next_iob!= "O":
                    bilou_list.append("I-" + ne_tags[i][2:])
                prev_iob = curr_iob

            train_id2sent[seed_id] = sentence
            train_id2pos[seed_id] = (" ").join(pos_list)
            train_id2ner[seed_id] = (" ").join(ner_list)
            train_id2nerBILOU[seed_id] = (" ").join(bilou_list)
            train_id2arg2rel[seed_id] = relation_dict
            seed_id = seed_id + 1
        return train_id2sent,train_id2pos, train_id2ner,train_id2nerBILOU,train_id2arg2rel


