import os
import sys

if os.name == 'nt':
    module_path = os.path.abspath(os.path.join('..\..\..'))
else:
    module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from entity_linkage.typing.entity_typing import entity_typing
from entity_linkage.typing.oneShotRelationalLearning import data
from entity_linkage.typing.oneShotRelationalLearning.args import read_options
from entity_linkage.typing.oneShotRelationalLearning.trainer import main
from entity_linkage.typing.oneShotRelationalLearning.trainer import get_scores
from shutil import copy2
import json

class OneShotRelationalLearning (entity_typing) :
    
    def read_dataset(self, file_names, options={}):
        """
        Reads a dataset in preparation for: train or test. Returns data in proper format for: train or test.
        Args:
            file_names: a list containing the fully qualified name to a file representing a knowledge graph containing triples like 
            concept:person:paavo_nurmi001   concept:personhasethnicity      concept:ethnicgroup:finnish.
            If using the papers dataset this is not required simply specify the directory of those files in future methods.
            options: Not used
            Returns:
                Creates a directory of files necessary for the future function calls to train, predict and evaluate. 
            Returns a directory the key being the directory name of the data.
        """
        # read in kb file_names[0]
        file_name = file_names[0]
        split_file = file_name[file_name.find('/') + 1 :]
        dir_name = "./data/" + split_file[: split_file.find('.')]
        if not os.path.exists (dir_name) :
            os.mkdir(dir_name)
        copy2(file_name, dir_name + "/path_graph")   
        triples = []
        with open(file_name) as input_file :
            for line in input_file :
                triples.append((line.split()[0], line.split()[1], line.split()[2]))
        relations = []
        entities = []
        rel2candidates = {}
        tasks = {} # key relation, triple as value
        for triple in triples : 
            entities.append(triple[0])
            relations.append(triple[1])
            entities.append(triple[2])
            if triple[1] not in rel2candidates :
                rel2candidates[triple[1]] = []
            if triple[0] not in rel2candidates[triple[1]] :
                rel2candidates[triple[1]].append(triple[0])
            if triple[1] not in tasks :
                tasks[triple[1]] = [] 
            tasks[triple[1]].append(triple) 
        
        train_tasks = {}
        test_tasks = {}
        dev_tasks = {}
        for count, rel in enumerate(relations) :
            if count % 6 == 0 :
                test_tasks[rel] = tasks[rel]
            elif count % 13 == 0 :
                dev_tasks[rel] = tasks[rel]
            else :
                train_tasks[rel] = tasks[rel]
    
        with open(dir_name + "/rel2candidates.json", "w") as rel_candidates :
            rel_candidates.write(json.dumps(rel2candidates))

        with open(dir_name + "/train_tasks.json", "w") as train_task_file :
            train_task_file.write(json.dumps(train_tasks))
        
        with open(dir_name + "/test_tasks.json", "w") as test_task_file :
            test_task_file.write(json.dumps(test_tasks))
        
        with open(dir_name + "/dev_tasks.json", "w") as dev_task_file :
            dev_task_file.write(json.dumps(dev_tasks))
            
        data.build_vocab(dir_name)
        data.for_filtering(dir_name, True)
        # create task files = full dataset split into dev, test, train sets. Dev can be empty(should be there because code needs it several places and I'm not 100% sure I understand how everywhere) split tasks 5-1 train/test 
        # path_graph is just the triples from the dataset
        # build_vocab (dataset(file_name pre .)) # need to previously mkdir and create path_graph file which can just be knowledge graph file 
        # output creates relation2ids and entity2ids
        # for_filtering (dataset, true) # saves e1rel_e2.json
        # e1rel_e2: e1+rel + : list of e2's concept:university:carnegie_mellon_universityconcept:organizationnamehasacronym": ["concept:organization:cmu"]  
        # tasks = full dataset split, key is relation, then list of lists each list representing a triple e1, r, e2 
        return dir_name


    def train(self, train_data, options={}):
        """
        Args:
            train_data: a path to the directory containing the data returned by read_dataset or from the original paper dataset
            options: Not used
        Returns:
            None: Saves training data in the model
        """
        parser = read_options(True)
        arguments = parser.parse_args(["--max_batches", "1000", "--random_embed", "--max_neighbor", "50", "--fine_tune", "--dataset", train_data])
        arguments.save_path = 'models/' + arguments.prefix

        main(arguments)
        # dupe read_options but manually update parse_args with the correct argument options
        # CUDA_VISIBLE_DEVICES=0 python trainer.py --random_embed --max_neighbor 50 --fine_tune --dataset train_data 
    

    def predict(self, test_data, model_details=None, options={}) :
        """
        Args:
            test_data: a path to the directory containing the data returned by read_dataset or directly included from original paper dataset
            model_details and options: Not Used
        Returns:
            A file containing a dictionary with two labeled lists of pairs: representing the supporting and query links present in the knowledge graph, these two sets combine to give us the evaluation metrics 
        """
        parser = read_options(True)
        arguments = parser.parse_args(["--random_embed", "--max_neighbor", "50", "--get_data", "--test", "--dataset", test_data])
        arguments.save_path = 'models/' + arguments.prefix
        query_pairs, support_pairs = main(arguments)
        result_json = {"query pairs" : query_pairs, "support pairs" : support_pairs}
        with open (test_data + "/result.txt", "w") as result_file :
            result_file.write(json.dumps(result_json))

        return test_data + "/result.txt"
        # dupe read_options but manually update parse_args with the correct argument options including test and test_data
        # CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50 --test --dataset test_data


    def evaluate(self, test_data, prediction_data=None , options={}) :
        """
        Args:
            test_data: a path to the directory containing the data returned by read_dataset or directly from the original paper dataset
            prediction_data: If present runs the evaluation on this data, if not then will run predict first before evaluating.
            prediction_data is in the format (Query_pairs, Support_pairs) both pairs are lists
            options: Not Used
        Returns:
            A list containing the MRR(Mean Reciprocal Rank) found, and an F1 score of 0 because it is not a metric for this paper.
        """
        parser = read_options(True)
        results = None
        if prediction_data != None :
            prediction_json = json.loads(prediction_data)
            query_pairs = prediction_data["query pairs"]
            support_pairs = prediction_data["support pairs"]
            results = get_scores(args, query_pairs, support_pairs)
        else :
            arguments = parser.parse_args(["--random_embed", "--max_neighbor", "50", "--test", "--dataset", test_data])
            arguments.save_path = 'models/' + arguments.prefix
            results = main(arguments)
        
        with open (test_data + "/eval_result.txt", "w") as result_file :
            result_file.write("HITS10 = " + str(results[0]) + "\n")
            result_file.write("HITS5 = " + str(results[1]) + "\n")
            result_file.write("MRR = " + str(results[2]) + "\n")

        return test_data + "/eval_result.txt"

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass
#Implementation of 
#@inproceedings{Xiong_Oneshot,
#  author    = {Wenhan Xiong and
#               Mo Yu and
#               Shiyu Chang and
#               Xiaoxiao Guo and
#               William Yang Wang},
#  title     = {One-Shot Relational Learning for Knowledge Graphs},
#  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
#               Language Processing, Brussels, Belgium, October 31 - November 4, 2018},
#  pages     = {1980--1990},
#  publisher = {Association for Computational Linguistics},
#  year      = {2018},
#  url       = {https://aclanthology.info/papers/D18-1223/d18-1223},
#  timestamp = {Sat, 27 Oct 2018 20:04:50 +0200},
#}
