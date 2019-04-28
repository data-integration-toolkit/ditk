from linguistic_ner import LinguisticRNN

class LingusticStructureforNER():
    """
        Wrapper implementation for the paper -
        Leveraging Linguistic Structures for Named Entity Recognition with Bidirectional Recursive Neural Networks
            - Li, Peng-Hsuan  and  Dong, Ruo-Ping  and  Wang, Yu-Siang  and  Chou, Ju-Chieh  and  Ma, Wei-Yun
    """

    def __init__(self):
        self.processor = LinguisticRNN()

    def read_dataset(self, file_dict, dataset_name=None, *args, **kwargs):

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

        self.processor.process_dataset(data, **kwargs)
        return data

    def train(self, data, *args, **kwargs):
        """
        Train the specified model on the given dataset.

        :param data: A dictionary of lists from the read_dataset() method.
        :param args:
        :param kwargs:
                    embedding_files: list
                            List of embedding files required by the training module

        :return: None (Stores the model instance in the program)
        """
        try:
            pretrain = False
            if 'pretrain' in kwargs and kwargs['pretrain'] in (True, False):
                pretrain = kwargs['pretrain']
            self.processor.train_model(data, pretrain, **kwargs)
        except Exception as e:
            print('Error while training: ', str(e))

    def predict(self, data, *args, **kwargs):
        """
        Annotates the text with the entity type tags and mentions.

        :param data: iterable
                    Test data in an iterable format from read_dataset()
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
        formatted_data = None
        try:
            formatted_data = []
            for line in data:
                if len(line) > 2:
                    formatted_data.append((None, None, line[0], line[3]))
                else:
                    if len(formatted_data) != 0 and len(formatted_data[-1]) != 0:
                        formatted_data.append(())

            formatted_data.append(())
        except Exception as e:
            print('Error while converting data: ', str(e))
        return formatted_data

    def evaluate(self, predictions, groundTruths, *args,
                 **kwargs):
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
        score = (0, 0, 0)
        try:
            score = self.processor.evaluate(predictions, groundTruths)
        except Exception as e:
            print('Error while evaluating: ', str(e))
        return score

    def save_model(self, location='models'):
        try:
            self.processor.save_model(location)
        except Exception as e:
            print('Error while saving model: ', str(e))

    def load_model(self, location='models'):
        try:
            self.processor.load_model(location)
        except Exception as e:
            print('Error while loading model: ', str(e))
