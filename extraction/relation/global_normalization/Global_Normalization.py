from relation_extraction_2 import RelationExtractionModel
from createDataStream_setup1 import DataStream_Setup1
from createDataStream_setup2 import DataStream_Setup2
from train import TrainModel
class Global_Normalization(RelationExtractionModel):

    def read_dataset(self, input_file, dataset_name, config_file, setupOption, *args, **kwargs):
        """

        1. Reads the dataset from input_file location
        2. It uses H5PYDataset library to split the data for train, test and dev and create a fuel dataset for each of them.

        Args:
            input_file: Filepath with list of files to be read
        Returns:
            Separate H5DY dataset created for train, dev and test

        """
        if(setupOption == 1):
            data_stream = DataStream_Setup1(config_file)
        else:
            data_stream = DataStream_Setup2(config_file)

        return data_stream.load_dataset(input_file, dataset_name)

    def data_preprocess(self, input_data, *args, **kwargs):
        pass

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

    def train(self, train_data, database_name, configfile,test_id2sent, test_id2arg2rel, *args, **kwargs):
        """

        Trains a model on the given training data

        1. Reads the pre-trained embedding created
        2. Uses theano to train the model

        Args:
            train_data: post-processed data to be trained.

        Returns:
            Trained model.
        """
        return TrainModel().train(train_data,database_name, configfile,test_id2sent, test_id2arg2rel)

    def predict(self, test_data, entity_1=None, entity_2=None, trained_model=None, *args, **kwargs):
        """

        Predict the relation between entity_1 and entity_2 from the input sentence using the trained_model
        Args:
            test_data: Sentence (non-tokenize).
            entity_1, entity_2 : Two entities
            trained_model: the trained model from the method - def train().
        Returns:
            relation: entity type and relation between them

        """
        pass

    def evaluate(self, input_data, trained_model=None, *args, **kwargs):
        """

        Evaluates the result based on the test dataset and the evauation metrics  [Precision,Recall,F1, or others...]
         Args:
             input_data: Test dataset
             trained_model: trained model created using the method: train(train_data)
        Returns:
            performance metrics: tuple with (precision,recall,f1)

        """
        pass
