import abc

class RelationExtraction(abc.ABC):

    def __init__(self):
        pass

    @classmethod
    @abc.abstractmethod
    def read_dataset(self, input_file, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def data_preprocess(self, input_data, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def tokenize(self, input_data, ngram_size=None, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def train(self, train_data, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def predict(self, test_data, entity_1=None, entity_2=None, trained_model=None, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def evaluate(self, input_data, trained_model=None, *args, **kwargs):
        pass