import unittest
from ned_with_web_links import EntityNormalization


class TestEntityNormalization(unittest.TestCase):

    def setUp(self):
        # instantiate the implemented class
        self.entity_normalization = EntityNormalization()
        self.model, train_set, eval_set, test_set = self.load_model()
        self.test_list = test_set

    def test_predict(self):
        # test if predicts returns at least a list with two elements
        results = self.entity_normalization.predict(self.model, self.test_list)
        print results

    def load_model(self):
        train_set, eval_set, test_set = \
            self.entity_normalization.read_dataset("/data0/linking/wikipedia/dumps/20150901/", (0.8, 0.1, 0.1))
        return self.entity_normalization.train(train_set), train_set, eval_set, test_set


if __name__ == '__main__':
    unittest.main()