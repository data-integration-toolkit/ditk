from experiment import Experiment
import utils
# from ner import Ner
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from ner import Ner

DATASET_NAMES = 'dataset_names'
ENTITY_LIST = 'entity_list'
ANNOTATION_FORMAT = 'annotation_format'
EMBEDDING_FILES = 'embedding_files'


class TransferLearningforNER(Ner):
    """
        Wrapper implementation for the paper -
        Transfer Learning for Entity Recognition of Novel Classes
            - Juan Diego Rodriguez, Adam Caldwell and Alexander Liu
    """

    def __init__(self):
        self.processor = Experiment()

    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
        """
        Reads the specified dataset files required by training, prediction, evaluation module.

        :param file_dict: dictionary

                 {
                    "train": dict, {key="file description":value="file location"},
                    "dev" : dict, {key="file description":value="file location"},
                    "test" : dict, {key="file description":value="file location"},
                 }
        :param dataset_name: str
                Name of the dataset required for calling appropriate utils, converters
        :param args:
        :param kwargs:
                        entity_list: list
                                List of entities annotated by the target dataset

                        annotation_format: str
                                *IOB1
                                *IOB2

                        embedding_files: dictionary
                            Dictionary of locations of embedding files, modules required by the code



        :return:    dictionary
                    A dictionary of iterables containing the files in required by the training format.
                    {
                        'train' : iterable,
                        'test' : iterable,
                        'dev' : iterable
                    }
        """

        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()
                processed_data = []
                for line in raw_data:
                    if len(line.strip()) > 0:
                        if len(line.split()) < 8:
                            continue
                        processed_data.append(line.strip().split())
                    else:
                        if len(processed_data) > 0:
                            processed_data.append(list(line))

                data[split] = processed_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)

        entity_list = None if ENTITY_LIST not in kwargs else kwargs[ENTITY_LIST]
        annotation_format = 'IOB1' if ANNOTATION_FORMAT not in kwargs else kwargs[ANNOTATION_FORMAT]
        embedding_files = None if EMBEDDING_FILES not in kwargs else kwargs[EMBEDDING_FILES]
        self.processor.prep_data(data,dataset_name, embedding_files, entity_list, annotation_format)

        return data

    def train(self, data, *args, **kwargs):
        """
        Train the specified model on the given dataset.

        :param data: A dictionary of lists from the read_dataset method.
        :param args:
        :param kwargs:
                        shuffle_seed: str (Seed value for dataset shuffle)
                        train_length: str (Length of the target dataset to be used for transfer learning)
                        classifier: str
                            * CRF: the CRFTagger from nltk, which calls external CRFSuite.
                              averaged_perceptron : the averaged perceptron from nltk
                            * IIS: Improved Iterative Scaling, via nltk
                            * GIS: Generalized Iterative Scaling, via nltk
                            * naivebayes: Naive Bayes from nltk.

                        transfer_method: str
                            *tgt
                            *pred
                            *predCCA
                            *lstm

                        excluded_tags: list
                            List of tags which are not novel for the target dataset

        :return: None (Stores the model instance in the program)
        """
        try:
            self.processor.train(data, **kwargs)
        except Exception as e:
            print('Error while training: ', str(e))

    def predict(self, data, *args, **kwargs):
        """
        Annotates the text with the entity type tags and mentions.

        :param data: iterable
                    Test data in an iterable format from read_dataset
        :param args:
        :param kwargs:
        :return: list
                Returns the predicted entities with annotations as tuples in the list.
                e.g [(SpanStartIdx, SpanEndIdx, Entity1, PredTag), (SpanStartIdx, SpanEndIdx, Entity1, PredTag).....]
        """
        predictions = None
        try:
            predictions = self.processor.predict(data)
        except Exception as e:
            print('Error while predicting: ', str(e))
        return predictions

    def convert_ground_truth(self, data, *args, **kwargs):
        """
            Utility for converting data of actual test annotaions to prediction evaluation format.

        :return: list
                Converts the predicted annotations list into format of the actual test annotations.
        """
        formatted_actual = None
        try:
            corpora = self.processor.convert_data(data)
            corpora = utils.read_conll_ditk(corpora)
            actual = [sent for sent in corpora]

            formatted_actual = []
            for sent in actual:
                for token_data in sent:
                    formatted_actual.append((None, None, token_data[0][0], token_data[1]))
                formatted_actual.append(())
        except Exception as e:
            print('Error while converting data: ', str(e))
        return formatted_actual

    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        """

        Calculates evaluation metrics on chosen benchmark dataset [Precision,Recall,F1, or others...]
        
        :param predictions: list
                        List of predicted entity mentions from convert_ground_truth and entity types
        :param groundTruths: list
                        List of actual entity mentions from convert_ground_truth and entity types
        :param args:
        :param kwargs:
        :return: tuple
                    Returns the tuple (Precision, Recall, F1) for the selected dataset.
        """
        scores = (0, 0, 0)
        try:
            scores = self.processor.eval(predictions,groundTruths)
        except Exception as e:
            print('Error while evaluating: ', str(e))
        return scores

    def load_model(self, location=None):
        try:
            if not location:
                self.processor.load_model()
            else:
                self.processor.load_model(location)
        except Exception as e:
            print('Error while saving model: ', str(e))

    def save_model(self,location=None):
        try:
            if not location:
                self.processor.save_model()
            else:
                self.processor.save_model(location)
        except Exception as e:
            print('Error while loading model: ', str(e))
