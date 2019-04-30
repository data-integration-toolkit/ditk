from abc import ABCMeta, abstractmethod


class DITKModel:
    # Since SQL_NET is listed as group 13, I will just be handling my own modules, they are implemented in the child
    # classes respectively

    __metaclass__ = ABCMeta
    @classmethod
    @abstractmethod
    def extract_embedding(*args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def train(*args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def test(*args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def evaluate(*args, **kwargs):
        pass
