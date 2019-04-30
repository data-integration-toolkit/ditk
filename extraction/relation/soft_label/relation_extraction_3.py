# Python 3.x

import abc


class RelationExtractionModel(abc.ABC):

    def __init__(self):
        pass

    @classmethod
    @abc.abstractmethod
    def read_dataset(self, input_file, *args, **kwargs):
        """
        Reads a dataset to be used for training

 Note: The child file of each member overrides this function to read dataset
 according to their data format.

        Args:
                input_file: Filepath with list of files to be read
        Returns:
    (optional):Data from file
        """
        pass

    @classmethod
    @abc.abstractmethod
    def data_preprocess(self, input_data, *args, **kwargs):
        """
 (Optional): For members who do not need preprocessing. example: .pkl files
 A common function for a set of data cleaning techniques such as lemmatization, count vectorizer and so forth.
        Args:
                input_data: Raw data to tokenize
        Returns:
                Formatted data for further use.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def tokenize(self, input_data, ngram_size=None, *args, **kwargs):
        """
        Tokenizes dataset using Stanford Core NLP(Server/API)
        Args:
                input_data: str or [str] : data to tokenize
                ngram_size: mention the size of the token combinations, default to None
        Returns:
                tokenized version of data
        """
        pass

    @classmethod
    @abc.abstractmethod
    def train(self, train_data, *args, **kwargs):
        """
        Trains a model on the given training data

 Note: The child file of each member overrides this function to train data
 according to their algorithm.

        Args:
                train_data: post-processed data to be trained.

Returns:
                (Optional) : trained model in applicable formats.
             None: if the model is stored internally.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def predict(self, test_data, entity_1=None, entity_2=None,  trained_model=None, *args, **kwargs):
        """
        Predict on the trained model using test data
        Args:
      entity_1, entity_2: for some models, given an entity, give the relation most suitable
                test_data: test the model and predict the result.
                trained_model: the trained model from the method - def train().
                                          None if store trained model internally.
        Returns:
      probablities: which relation is more probable given entity1, entity2
          or
                relation: [tuple], list of tuples. (Eg - Entity 1, Relation, Entity 2) or in other format
        """
        pass

    @classmethod
    @abc.abstractmethod
    def evaluate(self, input_data, trained_model=None, *args, **kwargs):
        """
        Evaluates the result based on the benchmark dataset and the evauation metrics  [Precision,Recall,F1, or others...]
 Args:
     input_data: benchmark dataset/evaluation data
     trained_model: trained model or None if stored internally
        Returns:
                performance metrics: tuple with (p,r,f1) or similar...
        """
        pass
