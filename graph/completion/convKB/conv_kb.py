from graph.completion.graph_completion import GraphCompletion
from graph.completion.convKB import read_data
from graph.completion.convKB import train
from graph.completion.convKB import batching
from graph.completion.convKB import evaluation
from graph.completion.convKB import prediction
import numpy as np
import os, sys

class ConvKB(GraphCompletion):

    def read_dataset(self, fileName, options={}):
        """
        Reads a dataset in preparation for: train or test. Returns data in proper format for: train or test.

        Args:
            fileName: Name of file representing the dataset to read
            options: object to store any extra or implementation specific data

        Returns:
            Iterable data, optionally split into train, test, and possibly dev.
        """ 
        split_ratio = options.get("split_ratio", (0.7, 0.2, 0.1))
        self.embedding_dim = options.get("embedding_dim", 100)
        batch_size = options.get("batch_size", 128)

        train, dev, test, words_indexes, self.indexes_words, headTailSelector, \
            self.entity2id, self.id2entity, self.relation2id, self.id2relation, self.embedding \
                = read_data.read_data(fileName, split_ratio, self.embedding_dim)

        self.len_words_indexes = len(words_indexes)
        self.train_batch = batching.Batch_Loader(train, words_indexes, self.indexes_words, headTailSelector, \
                                   self.entity2id, self.id2entity, self.relation2id, self.id2relation, batch_size=batch_size)
        
        return train, dev, test


    def train(self, data, options={}):
        """
        Trains a model on the given input data

        Args:
            data: iterable of arbitrary format
            options: object to store any extra or implementation specific data

        Returns:
            ret: None. Trained model stored internally to instance's state. 
        """      
        options["data_size"] = len(data)
        options["sequence_length"] = np.array(list(data.keys())).astype(np.int32).shape[1]
        options["num_classes"] = np.array(list(data.values())).astype(np.float32).shape[1]
        options["len_words_indexes"] = self.len_words_indexes
        self.model = train.train(self.train_batch, self.embedding, self.embedding_dim, options)


    def predict(self, data, options={}):
        """
        Predicts on the given input data (e.g. knowledge graph). Assumes model has been trained with train()

        Args:
            data: iterable of arbitrary format. represents the data instances and features you use to make predictions
                Note that prediction requires trained model. Precondition: instance already stores trained model 
                information.
            options: object to store any extra or implementation specific data

        Returns:
            predictions: [tuple,...], i.e. list of predicted tuples. 
                Each tuple likely will follow format: (subject_entity, relation, object_entity), but isn't required.
        """  
        options["len_entity2id"] = len(self.entity2id)
        options["sequence_length"] = np.array(list(data.keys())).astype(np.int32).shape[1]
        options["num_classes"] = np.array(list(data.values())).astype(np.float32).shape[1]
        options["len_words_indexes"] = self.len_words_indexes
        entity_array = np.array(list(self.train_batch.indexes_ents.keys()))   

        result = prediction.predict(data, self.embedding, self.embedding_dim, entity_array, options) 

        output_result = []
        for e1,r,e2 in result:
            output_result.append((self.indexes_words[e1], self.indexes_words[r], self.indexes_words[e2]))

        ditk_path = ""
        for path in sys.path:
            if "ditk" in path and not "graph" in path:
                ditk_path = path
        
        output_file = ditk_path + '/graph/completion/convKB/result/output.txt'  
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            for e1, r, e2 in output_result:
                f.write(str(e1) + "\t" + str(r) + "\t" + str(e2))
                f.write('\n')

        return output_result


    def evaluate(self, benchmark_data, metrics={}, options={}):
        """
        Calculates evaluation metrics on chosen benchmark dataset.
        Precondition: model has been trained and predictions were generated from predict()

        Args:
            benchmark_data: Iterable testing split of dataset to evaluate on
            metrics: Dictionary of function pointers for desired evaluation metrics (e.g. F1, MRR, etc.)
                - Note: This abstract base class does not enforce a metric because some metrics are more appropriate 
                for a given benchmark than others. At least one metric should be specified
                - example format:
                    metrics = {
                        "F1": f1_eval_function,
                        "MRR": mrr_eval_function
                    }
            options: object to store any extra or implementation specific data

        Returns:
            evaluations: dictionary of scores with respect to chosen metrics
                - e.g.
                    evaluations = {
                        "f1": 0.5,
                        "MRR": 0.8
                    }
        """   
        options["len_entity2id"] = len(self.entity2id)
        options["sequence_length"] = np.array(list(benchmark_data.keys())).astype(np.int32).shape[1]
        options["num_classes"] = np.array(list(benchmark_data.values())).astype(np.float32).shape[1]
        options["len_words_indexes"] = self.len_words_indexes
        entity_array = np.array(list(self.train_batch.indexes_ents.keys()))
        metrics_list = evaluation.evaluate(benchmark_data, self.embedding, self.embedding_dim, entity_array, metrics, options)
        
        metrics_dic = {"AUC":None, "Average Precision":None, "MRR":None, "hit@10":None}
        metrics_dic["MRR"] = metrics_list[1]
        metrics_dic["hit@10"] = metrics_list[2]
        return metrics_dic

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass