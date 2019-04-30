from graph.embedding.analogy.dataset import Vocab, TripletDataset
from graph.embedding.analogy.analogy import ANALOGY
import unittest

# DATA_SET_TO_USE = 'FB15k'
DATA_SET_TO_USE = 'WN18'

# Model Parameters, using minimum parameters
EPOCH = 2
DIMENSIONS = 20  # remember since using hybrid ComplEx + DistMult actual embeddings are divided by 2

# validate and whole text omitted to save train time
TRAIN_FILE_NAME = "train.txt"
RELATIONS_FILE_NAME = "relation2id.txt"
ENTITIES_FILE_NAME = "entity2id.txt"
TEST_FILE_NAME = "test.txt"

if DATA_SET_TO_USE == 'FB15k':
    # For FB15k data set
    INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\FB15k\\"
    N_ENTITIES = 14951
    N_RELATIONS = 1345
    N_TRAIN = 483142
else:
    INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\WN18\\"
    N_ENTITIES = 40943
    N_RELATIONS = 18
    N_TRAIN = 141442


class TestAnalogyMethods(unittest.TestCase):

    # set up model for unit testing, since unit tests can be run in any order by test loader
    # will read and train data set in setUpClass and then perform tests
    @classmethod
    def setUpClass(cls):
        cls.graph_embedding = ANALOGY()  # initialize your Blocking method
        train_file_names = {"train": INPUT_FILE_DIRECTORY + TRAIN_FILE_NAME,
                            "relations": INPUT_FILE_DIRECTORY + RELATIONS_FILE_NAME,
                            "entities": INPUT_FILE_DIRECTORY + ENTITIES_FILE_NAME}
        # Read the data set
        cls.graph_embedding.read_dataset(train_file_names)
        parameters = {"epoch": EPOCH, "dim": DIMENSIONS}
        # learn the embeddings
        cls.graph_embedding.learn_embeddings(parameters)

    def setUp(self):
        pass

    def test_read_dataset(self):
        # check to make sure required data is populated and count is correct
        self.assertTrue((type(self.graph_embedding.train_dat) is TripletDataset))
        self.assertTrue(len(self.graph_embedding.train_dat) == N_TRAIN)
        self.assertTrue(type(self.graph_embedding.ent_vocab) is Vocab)
        self.assertTrue(self.graph_embedding.n_entity == N_ENTITIES)
        self.assertTrue(type(self.graph_embedding.rel_vocab) is Vocab)
        self.assertTrue(self.graph_embedding.n_relation == N_RELATIONS)

    def test_learn_embeddings(self):
        # check embeddings
        # each embedding has real and imaginary parts in addition to plain embedding for DistMult part

        # check entities
        e = self.graph_embedding.model.params['e'].data
        e_re = self.graph_embedding.model.params['e_re'].data
        e_im = self.graph_embedding.model.params['e_im'].data
        self.assertEqual(e.shape, (N_ENTITIES, DIMENSIONS/2))
        self.assertEqual(e_re.shape, (N_ENTITIES, DIMENSIONS/2))
        self.assertEqual(e_im.shape, (N_ENTITIES, DIMENSIONS/2))

        # check relations
        r = self.graph_embedding.model.params['r'].data
        r_re = self.graph_embedding.model.params['r_re'].data
        r_im = self.graph_embedding.model.params['r_im'].data
        self.assertEqual(r.shape, (N_RELATIONS, DIMENSIONS/2))
        self.assertEqual(r_re.shape, (N_RELATIONS, DIMENSIONS/2))
        self.assertEqual(r_im.shape, (N_RELATIONS, DIMENSIONS/2))

    def test_evaluate(self):
        evaluate_file_names = {"test": INPUT_FILE_DIRECTORY + TEST_FILE_NAME}
        evaluations = self.graph_embedding.evaluate(evaluate_file_names)
        # Evaluations could be a dictionary or a sequence of metrics names
        self.assertIsInstance(evaluations, dict)
        self.assertIn("MRR", evaluations)
        self.assertIn("MRR(filter)", evaluations)
        self.assertIn("Hits@1", evaluations)
        self.assertIn("Hits@3", evaluations)
        self.assertIn("Hits@10", evaluations)
        self.assertIn("Hits@1(filter)", evaluations)
        self.assertIn("Hits@3(filter)", evaluations)
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
    unittest.main(verbosity=9)
