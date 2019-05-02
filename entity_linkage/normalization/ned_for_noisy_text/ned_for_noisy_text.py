import abc
from src.Experiment import Experiment


class EntityNormalization(abc.ABC):
    # Any shared data strcutures or methods should be defined as part of the parent class.
    # A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
    # The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group.

    experiment = None

    @classmethod
    @abc.abstractmethod
    def read_dataset(cls, dataset_name, split_ratio, *args):
        '''
        :param dataset_name: name of dataset
        :param split_ratio: (train_ratio, validation_ration, test_ratio)
        :param kwargs: other parameters for specific model (optional)
        :return: train_data, valid_data, test_data
        '''
        return

    @classmethod
    @abc.abstractmethod
    def train(cls, train_set):
        '''
        :param train_set: train dataset
        :return: trained model
        '''
        experiment = Experiment("user", "1234", 'NEDforNoisyText', 'localhost', train_set)
        for x in xrange(8):
            experiment.train(model_name='.' + str(x))
        return experiment


    @classmethod
    @abc.abstractmethod
    def predict(cls, model, test_set):
        '''
        :param model: a trained model
        :param test_set: a list of test data
        :return: a list of prediction, each item with the format
        (entity_name, wikipedia_url(optional), geolocation_url(optional), geolocation_boundary(optional))
        '''
        model.evaluate()
        return

    @classmethod
    @abc.abstractmethod
    def evaluate(cls, model, eval_set):
        '''
        :param model: a trained model
        :param eval_set: a list of validation data
        :return: (precision, recall, f1 score)
        '''
        # clear existing links
        model.evaluate()
        return