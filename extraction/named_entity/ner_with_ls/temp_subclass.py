import preprocess
import model
import tensorflow as tf
import numpy as np
from batcher import Batcher
import sys, pyhocon, os
from model import Model

from extraction.named_entity.ner import Ner

class NER_with_LS:
    def __init__(self, dataset_name):
        self.config = pyhocon.ConfigFactory.parse_file("experiments.conf")[dataset_name]
        self.dataset_name = dataset_name
        self.ground_truth_labels = None
        self.saver = None

    def read_dataset(self, file_dict) :
        data_dict = preprocess.read_data(self.dataset_name)
        return data_dict

    def split_data_txt(self, input_file_path, file_dict, options): # train.txt, dev.txt, test.txt

        split_ratio = options.get("ratio", (0.7, 0.15, 0.15)) 

        docs = []
        with open(input_file_path, 'r') as f:
            for line in f:
                docs.append((line.strip()))

        train_dev_split_idx = int(len(docs) * split_ratio[0])
        dev_test_split_idx = int(len(docs) * (split_ratio[0] + split_ratio[1]))

        data = {}
        data["train"] = docs[:train_dev_split_idx]
        data["dev"] = docs[train_dev_split_idx:dev_test_split_idx]
        data["test"] = docs[dev_test_split_idx:]

        for key in data:
            with open(file_dict[key], 'w') as f:
                for line in data[key]:
                    f.write(line+"\n")

        print("=== Splited! ")

    def train(self, data):
        # print("len of train_data: ", len(data["train"]))
        # print(data["train"])
        if len(data["train"]) <= 2 and len(data["dev"]) <= 2:
            print("train data size is not enough...")
        else:
            model.add_train(data, self.config)
        return

    def predict(self, test_data):
        pred_lables = []
        if len(test_data) <= 2:
            print("test data size is not enough...")
        else:
            pred_lables, ground_truth = model.add_predict(self.config, test_data)
            self.ground_truth_labels = ground_truth

        return pred_lables

    def convert_ground_truth(self, test_data):  # inputs are not necessary
        return self.ground_truth_labels

    def evaluate(self, pred_labels, ground_truth_labels): # inputs are not necessary
        scores = model.add_get_score( pred_labels, ground_truth_labels, self.config, self.dataset_name)
        return scores


    def save_model(self, file=None):
        model.save_model(self.config)


    def load_model(self, file=None):
        model.load_model(self.config)



    

