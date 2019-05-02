import subprocess
import os
from .utils import *
from graph.completion.graph_completion import GraphCompletion
from graph.completion.deepPath import sl_policy 
from graph.completion.deepPath import policy_agent
from graph.completion.deepPath import evaluate
from shutil import copy2
import copy

class DeepPath(GraphCompletion) :
    def read_dataset(self, file, options={}):
        """
        Reads a dataset in preparation for: train and test. Returns data in proper format for: both train and test.
        Args:
            file: a path to a file, containing knowledge graph data as triples like(concept:color:mixture concept:thinghascolor concept:color:green)
            options: Not used 
        Returns:
            A dictionary, containing the directory name as the key. And the relations to be trained on as the value 
        Raises:
            None
        """
        # open dataset files and does preprocessing
        split_file = file[file.find('/') + 1:]
        dir_name = split_file[: split_file.find('.')]
        if not os.path.exists(dir_name) :
            os.mkdir(dir_name)
        copy2(file, dir_name + "/raw.kb")
        
        triples = []
        with open(file) as input_file :
            for line in input_file :
                triples.append((line.split()[0].replace(":", "_"), line.split()[1], line.split()[2].replace(":","_")))
        entities = []
        relations = []
        inverse_triples = []
        entity_relation = {} # key = relation, entity = list of entities associated with the relation
        relation_ranking = {}

        for triple in triples :
            if triple[1] == "generalizations" or triple[1] == "concept:haswikipediaurl" :
                continue
            entities.append(triple[0])
            relations.append(triple[1])
            relation = triple[1]
            if relation not in entity_relation :
                entity_relation[relation] = []
            if relation not in relation_ranking :
                relation_ranking[relation] = 1
            else :
                relation_ranking[relation] += 1
            entity_relation[relation].append(triple[2])
            entities.append(triple[2])
            inv_relation = triple[1] + "_inv"
            relations.append(inv_relation)
            inverse_triples.append((triple[2], triple[1] + "_inv", triple[0]))

        all_triples = copy.deepcopy(triples)
        all_triples = all_triples + inverse_triples
        with open (dir_name + "/kb_env_rl.txt", "w") as f :
            for triple in all_triples :
                f.write(str(triple[0]) + " " + str(triple[2]) + " " + str(triple[1]) + "\n")
        
        entity_id = 0 
        entity_to_id = {}
        for entity in entities :
            if entity not in entity_to_id :
                entity_to_id[entity] = entity_id
                entity_id += 1
        
        relation_id = 0 
        relation_to_id = {}
        for relation in relations :
            if relation not in relation_to_id :
                relation_to_id[relation] = relation_id
                relation_id += 1

        train_id_triples = []
        train_pos = {}
        relation_count = {}
        for triple in all_triples :
            if triple[1] not in train_pos :
                train_pos[triple[1]] = []
            train_pos[triple[1]].append((triple[0], triple[2], triple[1]))
            e1 = entity_to_id[triple[0]]
            e2 = entity_to_id[triple[2]]
            r = relation_to_id[triple[1]]
            if r not in relation_count :
                relation_count[triple[1]] = 0
            relation_count[triple[1]] += 1
            train_id_triples.append((e1, e2, r))

        with open (dir_name + "/entity2id.txt", "w") as entity_to_ids :
            entity_to_ids.write(str(len(list(entity_to_id.keys()))) + "\n")
            for entity_name in list(entity_to_id.keys()) :
                entity_to_ids.write(entity_name + "\t" + str(entity_to_id[entity_name]) + "\n")
        with open (dir_name + "/relation2id.txt", "w") as relation_to_ids :
            relation_to_ids.write(str(len(list(relation_to_id.keys()))) + "\n")
            for relation_name in list(relation_to_id.keys()) :
                relation_to_ids.write(relation_name + "\t" + str(relation_to_id[relation_name]) + "\n")
        with open (dir_name + "/train2id.txt", "w") as train_to_ids :
            train_to_ids.write(str(len(train_id_triples)) + "\n")
            for triple in train_id_triples :
                train_to_ids.write(str(triple[0]) + " " + str(triple[1]) + " " + str(triple[2]) + "\n")

        # transX is a function used to get the KG embeddings necessary for training built from the cpp code found in https://github.com/thunlp/Fast-TransX 
        # subprocess.call("g++ transE.cpp -o transX -pthread -O3 -march=native")
        transX_output = subprocess.call(["./transX", "-input", dir_name + "/", "-output", dir_name + "/"])
            
        os.rename(dir_name + "/relation2vec.vec", dir_name + "/relation2vec.bern")
        os.rename(dir_name + "/entity2vec.vec", dir_name + "/entity2vec.bern")

        all_with_truth = {}
        for triple in triples :
            if triple[1] not in all_with_truth :
                all_with_truth[triple[1]] = []
            all_with_truth[triple[1]].append("$thing?" + triple[0] + "," + "$thing?" + triple[2] + ": +")
            for entity in entity_relation[triple[1]] :
                if entity == triple[2] :
                    continue
                all_with_truth[triple[1]].append("$thing?" + triple[0] + "," + "$thing?" + entity + ": -")
        train_with_truth = {}
        test_with_truth = {}
        for relation in relations :
            if relation.endswith("_inv") :
                continue
            counter = 0
            for elem in all_with_truth[relation] :
                if counter % 4 == 0 or counter == 0 :
                    if relation not in test_with_truth :
                        test_with_truth[relation] = []
                    test_with_truth[relation].append(elem)
                else :
                    if relation not in train_with_truth :
                        train_with_truth[relation] = []
                    train_with_truth[relation].append(elem)

        if not os.path.exists(dir_name + "/tasks") :
            os.mkdir(dir_name + "/tasks")
        task_relations = []
        for relation in set(relations) :
            if "_inv" in relation or (relation not in test_with_truth and relation not in train_with_truth) : 
                continue
            task_relation = relation.replace(":", "_")
            task_relations.append(task_relation)
            if not os.path.exists (dir_name + "/tasks/" + task_relation) :
                os.mkdir(dir_name + "/tasks/" + task_relation)
            copy2(dir_name + "/kb_env_rl.txt", dir_name + "/tasks/" + task_relation + "/graph.txt")
            with open (dir_name + "/tasks/" + task_relation + "/test.pairs", "w") as test_pairs :
                if relation in test_with_truth :
                    for test_pair in test_with_truth[relation] :
                        test_pairs.write(test_pair + "\n")
                    if len(test_with_truth[relation]) == 0 :
                        continue
                    sort_test = list(sorted(test_with_truth[relation]))
                    with open (dir_name + "/tasks/" + task_relation + "/sort_test.pairs", "w") as sort_file :
                        for sorted_pair in sort_test :
                            sort_file.write(sorted_pair + "\n")
            with open (dir_name + "/tasks/" + task_relation + "/train.pairs", "w") as train_pairs :
                if relation in train_with_truth :
                    for train_pair in train_with_truth[relation] :
                        train_pairs.write(train_pair + "\n")
            with open (dir_name + "/tasks/" + task_relation + "/train_pos", "w") as train_pos_file :
                if relation in train_pos : 
                    for triple in train_pos[relation] :
                        train_pos_file.write(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")
            with open (dir_name + "/tasks/" + task_relation + "/path_stats.txt", "w") as stats_file :
                stats_file.write("")
        return_dict = {dir_name : relations}
        return return_dict


    def train (self, data, options={}): 
        """
        Trains a model on the given input data, which is a dictionary containing the data returned from read_dataset
        Args:
            data: a dictionary containing the information returned in read_dataset
            options: can contain an optional relation to only train on that relation. 
        Returns:
            None. Trained model stored internally. 
        Raises:
            None
        """
        dir_name = list(data.keys())[0]
        relations = list(data.values())[0]
        if "relation" in options :
            relations = [options["relation"]]

        set_data_path(dir_name)
        for relation in relations :
            if relation.endswith("_inv") :
                continue
            relation = relation.replace(":", "_")
            sl_policy.setup(relation, dir_name)
            sl_policy.train(relation)
            policy_agent.setup(relation, dir_name, "retrain()")
            policy_agent.retrain(relation)


    def predict(self, data, options={}):
        """
        Predicts on the given input data (e.g. knowledge graph). Assumes model has been trained with train()
        Args:
            data: a dictionary containing the data returned from read_dataset, the directory name the data contains and the relations 
            we will be training on as the value
            options: optionally contains a relation to enable predicting on a single relation type rather then all of them. 
        Returns :
            a list of paths to use in the full graph. 
            eg: concept:athleteplayssport -> concept:teamplayssport_inv -> concept:teamplaysincity -> concept:stadiumlocatedincity_inv -> concept:leaguestadiums_inv 
        """
        # same as train except run python policy_agent.py $relation test
        dir_name = list(data.keys())[0]
        relations = data[dir_name]
        if "relation" in options :
            relations = [options["relation"]]
            set_data_path(dir_name)
        relation_paths = []
        for relation in relations :
            if relation.endswith("_inv") :
                continue
            relation = relation.replace(":", "_")	    
            policy_agent.setup(relation, dir_name, "test()")
            relation_paths.append(policy_agent.test(relation))
        
        with open (dir_name + "/path_to_use.txt", 'w') as path_file :
            for relation_path in relation_paths :
                if len(relation_path) == 0 : 
                    continue
                for item in relation_path :
                    path_file.write(item[0] + '\n')
    
        return dir_name + "/path_to_use.txt"

    def evaluate(self, benchmark_data, metrics={}, options={}):
        """
        Calculates evaluation metrics on chosen benchmark dataset.
        
        Args:
            benchmark_data: a dictionary containing the test data returned from read_dataset
            options: Must contain the data returned from read_dataset, specifically the directory name to relations, 
            and a optionally contains a relation to enable evaluating by a particular relation rather then all of them together
            metrics: Dictionary of function pointers for desired evaluation metrics, valid currently is MAP for mean average precision the base metric used in the paper.
    
        
        Returns:
            evaluations: a dictionary of scores with respect to chosen metrics
        """
        dir_name = list(benchmark_data.keys())[0]
        relations = list(benchmark_data.values())[0]
        
        if "relation" in options :
            relations = [options["relation"]]
        
        precision_total = 0
        for relation in relations :
            if relation.endswith("_inv") :
                continue
            relation = relation.replace(":", "_")
            evaluate.setup(relation, dir_name)
            precision_total += evaluate.evaluate_logic()
        
        result = {}
        result["MAP"] = precision_total/len(relations)

        return result
    
    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass


# deeppath is the implementation of DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning 
# Citation: 
#   @InProceedings{wenhan_emnlp2017,
#   author    = {Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
#   title     = {DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning},
#   booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017)},
#   month     = {September},
#   year      = {2017},
#   address   = {Copenhagen, Denmark},
#   publisher = {ACL}
#   }

