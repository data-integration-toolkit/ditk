import ner_base
import os
import shutil
import sys
from D3NER import main
import csv
from collections import defaultdict
import subprocess
import numpy as np, pickle, re, os, subprocess, argparse
from D3NER.train.d3ner_model import BiLSTMCRF
from D3NER.train.dataset import BioCDataset
from D3NER.ner.data_utils import get_trimmed_glove_vectors
from collections import defaultdict
#from nltk.stem.snowball import SnowballStemmer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import linear_kernel
#from D3NER.ner.data_utils import load_vocab
#from D3NER.module.spacy import Spacy
#from D3NER.pre_process.segmenters.spacy import SpacySegmenter
#from D3NER.pre_process.tokenizers.spacy import SpacyTokenizer
#from D3NER.utils import Timer
#from D3NER.constants import ETYPE_MAP, ENTITY_TYPES
from D3NER import data_managers, readers, pre_process, ner, writers, constants
from D3NER import pipelines
from D3NER.pre_process import opt as pp_opt
from D3NER.ner import opt as ner_opt
conll_2000_tagset = {'NP', 'ADVP', 'ADJP', 'VP', 'PP', 'SBAR', 'CONJP', 'PRT', 'INTJ', 'LST', 'UCP'}

class DITKD3NER(ner_base.NER ):
    def __init__(self):
        self.dataset_name = "default"

    @staticmethod
    def convert_ground_truth(data, *args, **kwargs):
        annotation_file = data[0]
        text_file = data[1]
        id_dict = defaultdict(list)
        for line in open(text_file).readlines():
            id = line.split("\t")[0]
            id_dict[id] = []
            id_dict[id] = []

        for line in open(annotation_file).readlines():
            line = line.split("\t")
            value =(line[2],line[3],line[4],None,None)
            id_dict[line[0]].append(value)
        out = []
        for id, value in sorted(id_dict.items()):
            out.append(value)
        return out


    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
        limit = None
        if "limit" in kwargs:
           limit = kwargs["limit"]
        self.dataset_name = dataset_name
        out_dict = dict()
        #Parse to biocreative
        for key,value in file_dict.items():
            if isinstance(value,tuple):
                #Correct format
                out_dict[key] = value
            elif isinstance(value,str):
                if "ditk" in dataset_name.lower():
                    out_dict[key] = DITKD3NER.parse_conll_to_biocreative_format(DITKD3NER.ditk_to_conll2003(value))
                else:
                    out_dict[key] = DITKD3NER.parse_conll_to_biocreative_format(value)
            else:
                raise Exception("Unknown format: " + str(value))

        formatted_dict = dict()
        #Parse from biocreative to custom format
        for key,value in out_dict.items():
            if isinstance(value,tuple):
                #Correct format
                formatted_dict[key] = self.convert_annotations_and_tsv_to_format(value, dataset_name, key, limit=limit)
            else:
                raise Exception("Unknown format: " + str(value))
        return formatted_dict, out_dict

    def train(self, data, *args, **kwargs):
        train = BioCDataset(self.dataset_name, data["train"])
        dev = None
        if "dev" in data:
            dev = BioCDataset(self.dataset_name,data["dev"])
        embeddings = get_trimmed_glove_vectors(('D3NER/data/{}/embedding_data.npz').format(self.dataset_name))
        model = BiLSTMCRF(model_name=self.dataset_name, embeddings=embeddings, batch_size=120,
                          early_stopping=True,
                          display_step=True)
        model.load_data(train, dev=dev)
        model.build()
        #if not args.early_stopping:
        #    model.run_train(args.epoch, verbose=args.verbose)
        #else:
        model.run_train(100, verbose=True, patience=4)


    def predict(self, data, *args, **kwargs):
        #dataset = data
        #model = "pre_trained_models/chemdner/" + self.dataset_name
        pre_config = {pp_opt.SEGMENTER_KEY: pp_opt.SpacySegmenter(),
            pp_opt.TOKENIZER_KEY: pp_opt.SpacyTokenizer(),
            pp_opt.OPTION_KEY: [pp_opt.NumericNormalizer()]}
        nern_config = {ner_opt.NER_KEY: ner_opt.TensorNer(self.dataset_name, self.dataset_name)}
        reader = readers.BioCreativeReader(data)
        #writer = writers.BioCreativeWriter()
        data_manager = data_managers.CDRDataManager()
        pre_config = pre_config
        nern_config = nern_config
        raw_documents = reader.read()
        title_docs, abstract_docs = data_manager.parse_documents(raw_documents)
        title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
        abs_doc_objs = pre_process.process(abstract_docs,pre_config, constants.SENTENCE_TYPE_ABSTRACT)
        doc_objects = data_manager.merge_documents(title_doc_objs, abs_doc_objs)
        dict_nern = ner.process(doc_objects, nern_config)
        #writer.write(output_file, raw_documents, dict_nern)
        #print(dict_nern)
        out = []
        for key, value in sorted(dict_nern.items()):
            cur = []
            for entity in value:
                cur.append((entity.tokens[0].doc_offset[0], entity.tokens[-1].doc_offset[1], entity.content, entity.type))
            out.append(cur)
        return out


    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        true_positives = set()
        false_positives = set()
        false_negatives = set()
        for prediction, truth in zip(predictions, groundTruths):
            prediction_set = set(map(lambda x: (x[0],x[1],x[2]), prediction))
            truth_set = set(map(lambda x: (int(x[0]),int(x[1]),x[2]),truth))
            true_positives.update(prediction_set.intersection(truth_set))
            false_positives.update(prediction_set.difference(truth_set))
            false_negatives.update(truth_set.difference(prediction_set))
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return (precision, recall, f1)

    @staticmethod
    def write_ditk_output(predictions,data):
        annotatations_fname = data[0]
        text_fname = data[1]
        ground_truths = DITKD3NER.convert_ground_truth(data)
        text_dict = dict()
        with open(text_fname,"r") as text_f:
            for line in text_f.readlines():
                line_split = line.split("\t")
                text_dict[line_split[0] + "T"] = line_split[1]
                text_dict[line_split[0] + "A"] = line_split[2]

        folder = data[1].split("/")[:-1]
        outfile_name = "/".join(folder) + "/out.txt"
        sorted_items = sorted(text_dict.items())
        with open(outfile_name,"w") as f:
            for i in range(len(sorted_items)):
                prediction = predictions[i]
                ground_truth = ground_truths[i]
                text = sorted_items[i][1]
                offset = 0
                for word in text.split(" "):
                    predicted_tag = "O"
                    ground_truth_tag = "O"
                    for value in prediction:
                        if int(value[0]) == offset and value[2] == word:
                            predicted_tag = "I-MISC"
                    for value in ground_truth:
                        if int(value[0]) == offset:
                            ground_truth_tag = "I-MISC"
                    offset += len(word + " ")
                    f.write(word)
                    f.write(" ")
                    f.write(ground_truth_tag)
                    f.write(" ")
                    f.write(predicted_tag)
                    f.write("\n")
        return outfile_name






    @staticmethod
    def parse_conll_to_biocreative_format(data):
        #Format (id, start, end, entity, type)
        annotations = []
        #Format (id, text)
        texts = []
        id = 0
        if isinstance(data,str):
            with open(data,"r") as f:
                lines = list(f.readlines())
        else:
            lines = data
        cur_line = ""
        offset = 0
        in_title = True
        title = ""
        for i,line in enumerate(lines):
            if 'DOCSTART' not in line and not(i == len(lines) - 1):
                if isinstance(line,str):
                    split_line = line.split(" ")
                else:
                    split_line = line
                text = split_line[0] + " "
                if len(split_line) == 4:
                    if in_title:
                        title += text
                    else:
                        cur_line += text
                    if "B" in split_line[3] or "I" in split_line[3]:
                        annotations.append((id,in_title,offset,offset + len(text),text,split_line[3].split("-")[1].strip("\n")))
                    offset += len(text)
                if split_line[0] == ".":
                    in_title = False
            else:
                in_title = True
                texts.append((id, title, cur_line))
                cur_line = ""
                title = ""
                offset = 0
                id += 1
        folder = data.split("/")[:-1]
        annotations_fname = "/".join(folder) + "/annotations.tsv"
        texts_fname = "/".join(folder) + "/texts.txt"
        with open(annotations_fname,"w") as f:
            for annotation in annotations:
                f.write(str(annotation[0])) #ID
                f.write("\t")
                if annotation[1]:
                    f.write("T\t")
                else:
                    f.write("A\t")
                f.write(str(annotation[2])) #Start
                f.write("\t")
                f.write(str(annotation[3])) #end
                f.write("\t")
                f.write(annotation[4]) #entity
                f.write("\t")
                f.write(annotation[5]) #Type
                f.write("\n")
        with open(texts_fname,"w") as f:
            for text in texts:
                f.write(str(text[0]))
                f.write("\t")
                f.write(text[1])
                f.write("\t")
                f.write(text[2])
                f.write("\n")
        return (annotations_fname,texts_fname)

    @staticmethod
    def is_chunk_tag(chunk):
        if chunk == 'O' or str(chunk).startswith('B-') or str(chunk).startswith('I-'):
            return True
        return False
    @staticmethod
    def ditk_to_conll2003(input_file):
        with open(input_file,"r") as f:
                lines = list(f.readlines())
        converted_lines = []

        chunk_tag_format = None
        for i, line in enumerate(lines):
            if len(line.strip()) > 0:
                data = line.split()
                if chunk_tag_format is None:
                    chunk_tag_format = DITKD3NER.is_chunk_tag(data[2])

                if not chunk_tag_format:
                    tokens = str(data[2])
                    tokens = re.findall(r"[\w']+", tokens)
                    for token in tokens:
                        if token in conll_2000_tagset:
                            data[2] = 'I-' + token
                        else:
                            data[2] = 'O'

                converted_line = list()
                converted_line.extend(data[0:4])
                converted_lines.append(converted_line)

            else:
                converted_lines.append(line)
        folder = input_file.split("/")[:-1]
        fname = "/".join(folder) + "/tmp.txt"
        with open(fname,"w") as f:
            for line in converted_lines:
                if len(line) != 4:
                    f.write(line)
                else:
                    f.write(line[0].strip("/"))
                    f.write(" ")
                    f.write(line[1])
                    f.write(" ")
                    f.write(line[2])
                    f.write(" ")
                    f.write(line[3])
                    f.write("\n")

        return fname

    def convert_prediction_to_standard_format(self, predictions):
        return list(map(lambda p: (p[0],p[1],p[2], None, None),predictions))

    def convert_annotations_and_tsv_to_format(self, data, name, type, limit=None):
        annotations_f = open(data[0],"r")
        texts_f = open(data[1],"r")

        annotations_in = csv.reader(annotations_f,delimiter="\t")
        texts_in = csv.reader(texts_f,delimiter="\t")

        words = []

        key_set = set()
        #format is id -> (start, end, entity, type)
        annotations_dict = defaultdict(list)
        for row in annotations_in:
            key = row[0]
            key_set.add(key)
            annotations_dict[key].append((row[2],row[3],row[4],row[5]))

        #format id -> (title, body)
        texts_dict = dict()
        for row in texts_in:
            key = row[0]
            key_set.add(key)
            texts_dict[key] = (row[1],row[2])
            words.extend(row[1].split(" "))
            words.extend(row[2].split(" "))

        i = 0
        key_to_index_dict = dict()
        for key in sorted(key_set)[:limit]:
            key_to_index_dict[key] = i
            i += 1

        outfile_path = "D3NER/data/" + name
        if not os.path.exists(outfile_path):
            os.makedirs(outfile_path)
        outfile_name = "D3NER/data/" + name + "/" + name + "_" + type + ".txt"
        i =0
        with open(outfile_name,"w") as outfile:
            for key in list(texts_dict.keys()):
                try:
                    outfile.write(str(key_to_index_dict[key]))
                    outfile.write("|t|")
                    outfile.write(texts_dict[key][0])
                    outfile.write("\n")
                    outfile.write(str(key_to_index_dict[key]))
                    outfile.write("|a|")
                    outfile.write(texts_dict[key][1])
                    outfile.write("\n")
                    for annotation in annotations_dict[key]:
                        outfile.write(str(key_to_index_dict[key]))
                        outfile.write("\t")
                        outfile.write(annotation[0])
                        outfile.write("\t")
                        outfile.write(annotation[1])
                        outfile.write("\t")
                        outfile.write(annotation[2])
                        outfile.write("\t")
                        outfile.write(annotation[3])
                        outfile.write("\t")
                        outfile.write("D" + str(i))
                        outfile.write("\n")
                        i = i + 1
                    outfile.write("\n")
                except KeyError:
                    pass

        annotations_f.close()
        texts_f.close()

        files = ["all_chars.txt","all_words.txt","ab3p_tfidf.pickle","embedding_data.npz"]
        for file in files:
           shutil.copy2("D3NER/data/cdr/" + file,outfile_path + "/" +file)

        return outfile_name



    #def convert_conll_to_format(self,data):



if __name__ == "__main__":
    model = DITKD3NER()
    dataset = {"train": ("chemdner_corpus/training.annotations.txt", "chemdner_corpus/training.abstracts.txt"),
               "test": ("chemdner_corpus/evaluation.annotations.txt","chemdner_corpus/evaluation.abstracts.txt")}
    #dataset = {"train": ("cemp/train/chemdner_cemp_gold_standard_train.tsv","cemp/train/chemdner_patents_train_text.txt"),
    #           "test": ("cemp/test/chemdner_cemp_gold_standard_development_v03.tsv","cemp/test/chemdner_patents_development_text.txt")}
    #dataset = {"train": "conll/eng.train.txt", "test": "conll/eng.testa.txt"}
    dataset_converted, dataset_regular = model.read_dataset(dataset, 'cemp', limit=500)
    model.train(dataset_converted)
    predictions = model.predict(dataset_converted["test"])
    ground_truth = model.convert_ground_truth(dataset_regular["test"])
    print(model.evaluate(predictions,ground_truth))
