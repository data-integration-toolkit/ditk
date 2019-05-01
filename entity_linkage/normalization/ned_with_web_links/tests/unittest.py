import unittest
from main import EntityNormalization


class TestEntityNormalization(unittest.TestCase):

    def setUp(self):
        # instantiate the implemented class
        self.entity_normalization = EntityNormalization()
        self.input_file = 'sample_input.txt'
        self.model = self.load_model()
        self.test_list = []
        with open(self.input_file, "r") as file:
            for line in file:
                self.test_list.append(line)

    def test_predict(self):
        # test if predicts returns at least a list with two elements
        results = self.entity_normalization.predict(self.test_list)
        print results

    def load_model(self):
        return self.entity_normalization.train("/data0/linking/wikipedia/dumps/20150901/")


if __name__ == '__main__':
    unittest.main()