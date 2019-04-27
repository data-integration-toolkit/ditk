from graph.embedding.analogy.dataset import Vocab, TripletDataset
from graph.embedding.analogy.analogy import ANALOGY
import unittest

# For FB15k data set
INPUT_FILE_DIRECTORY = "D:\\USC\\CS548\\groupdat\\FB15k\\"
# validate and whole text omitted to save train time
TRAIN_FILE_NAME = "train.txt"
RELATIONS_FILE_NAME = "relation2id.txt"
ENTITIES_FILE_NAME = "entity2id.txt"
TEST_FILE_NAME = "test.txt"

N_ENTITIES = 0
N_RELATIONS = 0
N_TRAIN = 0

# Model Parameters, use minimum parameters
EPOCH = 2
DIMENSIONS = 20


class TestAnalogyMethods(unittest.TestCase):

    def setUp(self):
        self.graph_embedding = ANALOGY()  # initialize your Blocking method

    def t1_test_read_dataset(self):
        # set up input files
        train_file_names = {"train": INPUT_FILE_DIRECTORY + TRAIN_FILE_NAME,
                            "relations": INPUT_FILE_DIRECTORY + RELATIONS_FILE_NAME,
                            "entities": INPUT_FILE_DIRECTORY + ENTITIES_FILE_NAME}

        # Read the data set
        self.graph_embedding.read_dataset(train_file_names)

        # check to make sure required data is populated and count > 0
        self.assertTrue((type(self.graph_embedding.train_dat) is TripletDataset))
        self.assertTrue(len(self.graph_embedding.train_dat) > 0)
        self.assertTrue(type(self.graph_embedding.ent_vocab) is Vocab)
        self.assertTrue(self.graph_embedding.n_entity > 0)
        self.assertTrue(type(self.graph_embedding.rel_vocab) is Vocab)
        self.assertTrue(self.graph_embedding.n_relation > 0)

    def t2_test_learn_embeddings(self):

        # set model parameters
        parameters = {"epoch": EPOCH, "dim": DIMENSIONS}
        # learn the embeddings
        self.graph_embedding.learn_embeddings(parameters)
        # check embeddings
        print("  ")
        print(self.graph_embedding.model.params['e'].data)

    '''
        embedding_vector = np.array(output_vec)
        assertEquals(embedding_vector.shape[0],3)
    assertEquals(embedding_vector.shape[1],300) #Example: output vec should be 3 x 300

    def test_evaluate(self):
		evaluations = self.graph_embedding.evaluate()
		# Evaluations could be a dictionary or a sequence of metrics names
		self.assertIsInstance(evaluations, dict)
		self.assertIn("f1", evaluations)
		self.assertIn("MRR", evaluations)
		self.assertIn("Hits", evaluations)

		f1, mrr, hits = self.graph_embedding.evaluate()
		self.assertIsInstance(f1, float)
		self.assertIsInstance(mrr, float)
		if hits is not None:
            
        self.assertIsInstance(hits, float)
    '''


if __name__ == '__main__':
    loader = unittest.TestLoader()
    # ln = lambda f: getattr(TestAnalogyMethods, f).im_func.func_code.co_firstlineno
    ln = lambda f: getattr(TestAnalogyMethods, f).__code__.co_firstlineno
    lncomp = lambda a, b: ln(a) - ln(b)
    loader.sortTestMethodsUsing = lncomp
    unittest.main(testLoader=loader, verbosity=2)
