from os import system, getcwd

from extraction.named_entity.cvt.cvt import main_funct
from extraction.named_entity.cvt.preprocessing import main_funct_pre
from extraction.named_entity.cvt.ner import Ner


class CrossViewTraining(Ner):

    def __init__(self):
        self.data_dir = getcwd()+'/mini_data'
        self.output_train_filename = self.data_dir + "/raw_data/chunk/train.txt"
        self.output_test_filename = self.data_dir + "/raw_data/chunk/test.txt"
        self.data_size = 'mini'
        self.gdrive_mounted = 'f'
        self.ground_truth_dict = dict()

    def convert_ground_truth(self, data=None, *args, **kwargs):
        ground_truth = list()
        for line in data:
            if line:
                temp = line.split()
                token = temp[0]
                label = temp[3]
                ground_truth.append([None, None, token, label])
                self.ground_truth_dict[token] = label
            else:
                continue
        return ground_truth

    def read_dataset(self, file_dict=None, dataset_name=None, *args, **kwargs):
        """
        :param file_dict: dictionary
                    {
                        "train": "location_of_train",
                        "test": "location_of_test",
                        "dev": "location_of_dev",
                    }

        :param args:
        :param kwargs:
        :return: dictionary of iterables
                    Format:
                    {
                        "train":[
                                    [ Line 1 tokenized],
                                    [Line 2 tokenized],
                                    ...
                                    [Line n tokenized]
                                ],
                        "test": same as train,
                        "dev": same as train
                    }

        """
        standard_split = ["train", "test", "dev"]
        data = {}
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, mode='r', encoding='utf-8') as f:
                    raw_data = f.read().splitlines()
                data[split] = raw_data[2:]
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        train_lines = [" ".join(line.split()[0:4]) + "\n" for line in data['train']]
        test_lines = [" ".join(line.split()[0:4]) + "\n" for line in data['test']]
        with open(self.output_train_filename, 'w') as f:
            f.writelines(train_lines)
        with open(self.output_test_filename, 'w') as f:
            f.writelines(test_lines)
        del train_lines
        del test_lines
        system("rm -rf mini_data/models")
        system("rm -rf mini_data/preprocessed_data")
        main_funct_pre(data_dir=self.data_dir, size=self.data_size, gdrive_mounted='f')
        return data

    def train(self, data=None, *args, **kwargs):
        main_funct(mode='train', model_name='chunking_model', data_dir=self.data_dir, size=self.data_size,
                   gdrive_mounted=self.gdrive_mounted)

    def predict(self, data=None, *args, **kwargs):
        output_predictions = main_funct(mode='eval', model_name='chunking_model', data_dir=self.data_dir,
                                        size=self.data_size,
                                        gdrive_mounted=self.gdrive_mounted)
        output_file_path = self.data_dir + "/output.txt"
        pred_vals = list()
        with open(output_file_path, 'w') as pred_op_f:
            for val in output_predictions:
                for vall in val:
                    if vall[0] != '<missing>':
                        true_label = self.ground_truth_dict[vall[0]]
                        pred_op_f.write(vall[0] + ' ' + true_label + ' ' + vall[1] + '\n')
                        pred_vals.append([None, None, vall[0], vall[1]])
                pred_op_f.write('\n')
        return pred_vals

    def evaluate(self, predictions=None, ground_truths=None, *args, **kwargs):
        main_funct(mode='eval', model_name='chunking_model', data_dir=self.data_dir, size=self.data_size,
                   gdrive_mounted=self.gdrive_mounted)

    def save_model(self, file=None):

        """
        All the models get stored automatically after every 10 iterations.
        So no need to explicitly save any model. Always the recent most best performing model
        is saved as the top model to be used on load.

        :param file: Where to save the model - Optional function
        :return: None
        """
        return None

    def load_model(self, file=None):
        """
        The best model is automatically extracted on every run. The recent most best performing
        model is auto extracted. So no need to explicitly load any particular model.

        :param file: From where to load the model - Optional function
        :return: None
        """
        return None
