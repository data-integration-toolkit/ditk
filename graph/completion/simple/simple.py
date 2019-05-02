from trainer_tester import TrainerTester
from params import Params
from reader import *
from pprint import pprint
import sys
sys.path.append("..")
from graph_completion import graph_completion
import os


class SimplE(graph_completion):
    def __init__(self, path):
        model_name = "SimplE_ignr"
        datasets = ["wn18", "fb15k"]
        self.path = path
        data = datasets[0]

        params = Params()
        params.use_default(dataset=data, model=model_name) 
        self.params = params

        self.params.max_iterate = 200
        self.params.save_each = self.params.max_iterate / 4
        self.params.save_after = self.params.max_iterate / 4
        
        self.read_data = False
        self.trained = False
        self.predicted = False
        

    def read_dataset(self, fileName, options={}):
        """
        Reads and returns a dataset

        Args:
            fileName: Name of dataset to read
            options: object to store any extra or implementation specific data

        Returns:
            training, testing, and validation data
        """
        assert(fileName in ["fb15k", "wn18"])
        self.reader = Reader()
        self.reader.read_triples(os.path.join(self.path, fileName))

        self.dataset = fileName
        self.read_data = True

        return tuple(self.reader.triples.values())
        
    def train(self, data, options={}):
        """
        Trains a model on the given input data

        Args:
            data: [(subject, relation, object, ...)]
            options: object to store any extra or implementation specific data

        Returns:
            None. Generated embedding is stored in instance state. 
        """
        if not self.dataset:
            if 'dataset' in options:
                self.dataset = options['dataset']
            else:
                sys.exit("[ERR] Please specify dataset name (either 'fb15k' or 'wn18') in parameter options['dataset']")

        if not self.read_data:
            sys.exit("[ERR] Run read_dataset() before training")

        print("[!] Training...")

        tt = TrainerTester(model_name="SimplE_ignr", params=self.params, dataset=self.dataset, data=data, reader=self.reader)
        tt.train()
        self.coordinator = tt
        self.trained = True

    def predict(self, data, options={}):
        """
        Use generated embedding to predicts links the given input data (KG as list of triples).
        Assumes embedding has been generated model via train()

        Args:
            data: [(subject, relation, object, ...)]
            options: object to store any extra or implementation specific data

        Returns:
            predictions: [tuple,...], i.e. list of predicted tuples. 
                Each tuple will follow format: (subject_entity, relation, object_entity)
        """
        self.coordinator.model.reader.triples["test"] = data
        print("[!] Predicting links...")
        metrics, new_triples = self.coordinator.test_model_on(str(self.max_iterate))
        self.metrics = metrics

        return new_triples 

    def evaluate(self, benchmark_data, metrics, options={}):
        """
        Calculates evaluation metrics on chosen benchmark dataset.
        Precondition: model has been trained and predictions were generated from predict()

        Args:
            benchmark_data: [(subject, relation, object, ...)]
            metrics: List of metrics for desired evaluation metrics (e.g. hits1, hits10, mrr)
            options: object to store any extra or implementation specific data

        Returns:
            evaluations: dictionary of scores with respect to chosen metrics
                - e.g.
                    evaluations = {
                        "hits10": 0.5,
                        "mrr": 0.8
                    }
        """
        print("[!] Evaluating...")
        self.predict(benchmark_data)
        evals = {m:self.metrics[m] for m in metrics}
        if len(evals) != len(metrics):
            print("Please enter valid metrics: {mrr, hits1, hits10}")
        
        return evals

if __name__ == "__main__":
    s = SimplE("../data")
    datasets = ["fb15k", "wn18"]
    training, test, valid = s.read_dataset(datasets[0])
    s.train(training)
    s.predict(test)
    print(s.evaluate(valid, ["mrr", "hits10"]))