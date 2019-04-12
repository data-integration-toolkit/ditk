import abc

class EntityNormalization(abc.ABC):
    # Any shared data strcutures or methods should be defined as part of the parent class.
    # A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
    # The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group.

    @classmethod
    @abc.abstractmethod
    def read_dataset(cls, dataset_name: str, split_ratio: tuple, *args) -> tuple:
        '''
        :param dataset_name: name of dataset
        :param split_ratio: (train_ratio, validation_ration, test_ratio)
        :param kwargs: other parameters for specific model (optional)
        :return: train_data, valid_data, test_data
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def train(cls, train_set: list):
        '''
        :param train_set: a list of training data
        :return: trained model
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def predict(cls, model, test_set: list) -> list:
        '''
        :param model: a trained model
        :param test_set: a list of test data
        :return: a list of prediction, each item with the format
        (entity_name, wikipedia_url(optional), geolocation_url(optional), geolocation_boundary(optional))
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def evaluate(cls, model, eval_set: list) -> tuple:
        '''
        :param model: a trained model
        :param eval_set: a list of validation data
        :return: (precision, recall, f1 score)
        '''
        pass

    @abc.abstractmethod
    def save_model(self, file):
        """

        :param file: Where to save the model - Optional function
        :return:
        """
        pass

    @abc.abstractmethod
    def load_model(self, file):
        """

        :param file: From where to load the model - Optional function
        :return:
        """
        pass



