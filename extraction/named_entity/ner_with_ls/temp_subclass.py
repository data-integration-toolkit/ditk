
import preprocess
import model
import tensorflow as tf
import numpy as np
from batcher import Batcher
import sys, pyhocon, os

from model import Model

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
        train_file = self.config.raw_path+"/"+self.dataset_name+".train.txt"
        dev_file = self.config.raw_path+"/"+self.dataset_name+".dev.txt"

        if os.stat(train_file).st_size == 0 or os.stat(dev_file).st_size == 0:
            print("train data size is not enough...")
        else:
            print("train data size is enough to train...")
            model.add_train(data, self.config)

        return

    def predict(self, test_data):

        trained_model_exist = True
        if self.dataset_name == "conll":
            if not os.path.isfile("models/checkpoint"):
                trained_model_exist = False

        elif self.dataset_name == "ontonotes":
            if not os.path.isfile("models/checkpoint/tagging/checkpoint"):
                 trained_model_exist = False

        pred_lables = []
        test_file = self.config.raw_path+"/"+self.dataset_name+".test.txt"

        if os.stat(test_file).st_size == 0:
            print("test data size is not enough...")
        else:
            pred_lables, ground_truth = model.add_predict(self.config, test_data, trained_model_exist)
            self.ground_truth_labels = ground_truth

        return pred_lables

    def convert_ground_truth(self, test_data):  # inputs are not necessary

        if self.ground_truth_labels is None or len(self.ground_truth_labels)==0:
            print("There is no test data ... ")
            return []
        else:
            return self.ground_truth_labels

    def evaluate(self, pred_labels, ground_truth_labels): # inputs are not necessary

        scores = model.add_get_score( pred_labels, ground_truth_labels, self.config, self.dataset_name)
        return scores


    def save_model(self, file=None):
        trained_model_exist = True
        if self.dataset_name == "conll":
            if not os.path.isfile("models/checkpoint"):
                trained_model_exist = False

        elif self.dataset_name == "ontonotes":
            if not os.path.isfile("models/checkpoint/tagging/checkpoint"):
                 trained_model_exist = False

        model.save_model(self.config, trained_model_exist)


    def load_model(self, file=None):
        trained_model_exist = True
        if self.dataset_name == "conll":
            if not os.path.isfile("models/checkpoint"):
                trained_model_exist = False

        elif self.dataset_name == "ontonotes":
            if not os.path.isfile("models/checkpoint/tagging/checkpoint"):
                 trained_model_exist = False


        model.load_model(self.config, trained_model_exist)

