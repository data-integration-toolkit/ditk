from graph.embedding.sample_test import TestGraphEmbeddingMethods
from graph.embedding.analogy.analogy import ANALOGY
from graph.embedding.analogy.dataset import Vocab, TripletDataset
import unittest


class TestAnalogyMethods(TestGraphEmbeddingMethods):

    def setUp(self, input_file):
        self.graph_embedding = ANALOGY()  # initialize your Blocking method
        self.input_file = input_file
        # parse input_file into dictionary
        tl = []
        self.input_dict = {}
        with open(input_file) as f:
            for line in f.readlines():
                for token in line:
                    tl.append(token)
        self.input_dict["input_data_dir_path"] = tl[0]
        self.input_dict["train_file_name"] = tl[1]
        self.input_dict["num_train_entries"] = tl[2]
        self.input_dict["valid_file_name"] = tl[3]
        self.input_dict["num_valid_entries"] = tl[4]
        self.input_dict["test_file_name"] = tl[5]
        self.input_dict["num_test_entries"] = tl[6]
        self.input_dict["whole_text_filename"] = tl[7]
        self.input_dict["num_whole_text_entries"] = tl[8]
        self.input_dict["relations_filename"] = tl[9]
        self.input_dict["num_relations"] = tl[10]
        self.input_dict["entities_filename"] = tl[11]
        self.input_dict["num_entities"] = tl[12]
        self.input_dict["epochs"] = tl[13]
        self.input_dict["model_file_output_dir"] = tl[14]
        self.input_dict["metrics"] = tl[15]

    def test_read_dataset(self):
        file_names = {"train": self.input_dict["train_file_name"],
                      "valid": self.input_dict["valid_file_name"],
                      "whole": self.input_dict["whole_text_filename"],
                      "relations": self.input_dict["relations_filename"],
                      "entities": self.input_dict["entities_filename"]}

        self.graph_embedding.read_dataset(file_names)
        # If possible check if the read_dataset() function returns data of similar format
        # (e.g. vectors of any size, lists of lists, etc..)

        self.assertTrue(self.graph_embedding.train_dat, TripletDataset)  # assert non-empty list
        self.assertTrue(self.graph_embedding.valid_dat, TripletDataset)  # assert non-empty list
        self.assertTrue(self.graph_embedding.whole_graph, TripletDataset)  # assert non-empty list

        self.assertTrue(self.graph_embedding.rel_vocab, Vocab)  # assert non-empty list
        self.assertTrue(self.graph_embedding.ent_vocab, Vocab)  # assert non-empty list

    def test_learn_embeddings(self):
        # This fucntion could check on whether the embeddings were generated and if yes, then
        # it can check on whether the file exists
        pass

    def test_evaluate(self):
        evaluations = self.graph_embedding.evaluate()
        # Evaluations could be a dictionary or a sequence of metrics names
        self.assertIsInstance(evaluations, dict)
        self.assertIn("MRR", evaluations)
        self.assertIn("MRR(filter)", evaluations)
        self.assertIn("Hits@1", evaluations)
        self.assertIn("Hits@1(filter)", evaluations)
        self.assertIn("Hits@3", evaluations)
        self.assertIn("Hits@3(filter)", evaluations)
        self.assertIn("Hits@10", evaluations)
        self.assertIn("Hits@10(filter)", evaluations)

        self.assertIsInstance(evaluations["MRR"], float)
        self.assertIsInstance(evaluations["MRR(filter)"], float)
        self.assertIsInstance(evaluations["Hits@1"], float)
        self.assertIsInstance(evaluations["Hits@1(filter)"], float)
        self.assertIsInstance(evaluations["Hits@3"], float)
        self.assertIsInstance(evaluations["Hits@3(filter)"], float)
        self.assertIsInstance(evaluations["Hits@10"], float)
        self.assertIsInstance(evaluations["Hits@10(filter)"], float)


if __name__ == '__main__':
    unittest.main()
