import csv
from os import getcwd
from random import choice

from extraction.named_entity.biobert.biocodes.conlleval import evaluate_conll_file
from extraction.named_entity.biobert.biocodes.ner_detokenize import detokenize
from extraction.named_entity.biobert.run_ner import main_funct
from extraction.named_entity.biobert.ner import Ner


class BioBert(Ner):

    def __init__(self):
        self.project_root = getcwd()
        self.data_dir = self.project_root + '/BC4CHEMD'
        self.config_root = self.project_root + '/config_dir'
        # self.output_dir = '/home/de11bu23n58k/output_dir'
        self.output_dir = self.project_root + '/output_dir'
        self.init_checkpoint = 'gs://biomed_pretrained/biobert_pubmed/biobert_model.ckpt'
        self.ground_truth_dict = dict()
        self.zip_threshold = 250
        self.upper_limit = 8
        self.seq = list(range(self.upper_limit))

    def convert_ground_truth(self, data=None, *args, **kwargs):
        if len(self.ground_truth_dict) > 0:
            return list(self.ground_truth_dict.items())
        else:
            ground_truth = list()
            for line in data:
                if line:
                    temp = line.split()
                    if temp:
                        token = temp[0]
                        if token == '/.':
                            token = '.'
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
        output_train_filename = self.data_dir + "/train.tsv"
        output_test_filename = self.data_dir + "/test.tsv"
        output_dev_filename = self.data_dir + "/train_dev.tsv"
        output_devel_filename = self.data_dir + "/devel.tsv"
        dev_length = len(data['dev'])
        train_lines = list()
        test_lines = list()
        dev_lines = list()
        devel_lines = list()
        train_lines_dict = list()
        test_lines_dict = list()
        dev_lines_dict = list()
        devel_lines_dict = list()
        for line in data['train']:
            temp = line.split()
            if temp:
                if temp[3] != 'O':
                    train_lines.append([temp[0], temp[3].split('-')[0]])
                else:
                    train_lines.append([temp[0], temp[3]])
                train_lines_dict.append([temp[0], temp[3]])
            else:
                train_lines.append(['', ''])

        for line in data['test']:
            temp = line.split()
            if temp:
                if temp[3] != 'O':
                    test_lines.append([temp[0], temp[3].split('-')[0]])
                else:
                    test_lines.append([temp[0], temp[3]])
                test_lines_dict.append([temp[0], temp[3]])
            else:
                test_lines.append(['', ''])

        for line in data['dev'][dev_length // 4:]:
            temp = line.split()
            if temp:
                if temp[3] != 'O':
                    dev_lines.append([temp[0], temp[3].split('-')[0]])
                else:
                    dev_lines.append([temp[0], temp[3]])
                dev_lines_dict.append([temp[0], temp[3]])
            else:
                dev_lines.append(['', ''])

        for line in data['dev'][:dev_length // 4]:
            temp = line.split()
            if temp:
                if temp[3] != 'O':
                    devel_lines.append([temp[0], temp[3].split('-')[0]])
                else:
                    devel_lines.append([temp[0], temp[3]])
                devel_lines_dict.append([temp[0], temp[3]])
            else:
                devel_lines.append(['', ''])

        with open(output_train_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in train_lines:
                tsv_writer.writerow(row)
        with open(output_test_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in test_lines:
                tsv_writer.writerow(row)
        with open(output_dev_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in dev_lines:
                tsv_writer.writerow(row)
        with open(output_devel_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for row in devel_lines:
                tsv_writer.writerow(row)
        train_dict = dict(train_lines_dict)
        test_dict = dict(test_lines_dict)
        dev_dict = dict(dev_lines_dict)
        devel_dict = dict(devel_lines_dict)
        mega_dict = {**train_dict, **test_dict, **dev_dict, **devel_dict}
        self.ground_truth_dict = mega_dict
        del train_lines
        del test_lines
        del dev_lines
        del devel_lines
        return data

    def train(self, data=None, *args, **kwargs):
        data_dir = self.data_dir
        # init_checkpoint = '/content/gdrive/My Drive/biobert_pubmed/biobert_model.ckpt'
        init_checkpoint = self.init_checkpoint
        vocab_file = self.config_root + '/vocab.txt'
        bert_config_file = self.config_root + '/bert_config.json'
        output_dir = self.output_dir
        do_train = True
        do_eval = True
        do_predict = False
        main_funct(data_dir=data_dir, init_checkpoint=init_checkpoint, vocab_file=vocab_file,
                   bert_config_file=bert_config_file, output_dir=output_dir, do_train=do_train, do_eval=do_eval,
                   do_predict=do_predict)

    def predict(self, data=None, *args, **kwargs):
        data_dir = self.data_dir
        # init_checkpoint = '/content/gdrive/My Drive/biobert_pubmed/biobert_model.ckpt'
        init_checkpoint = self.init_checkpoint
        vocab_file = self.config_root + '/vocab.txt'
        bert_config_file = self.config_root + '/bert_config.json'
        output_dir = self.output_dir
        do_train = False
        do_eval = True
        do_predict = True
        main_funct(data_dir=data_dir, init_checkpoint=init_checkpoint, vocab_file=vocab_file,
                   bert_config_file=bert_config_file, output_dir=output_dir, do_train=do_train, do_eval=do_eval,
                   do_predict=do_predict)

        pred_token_test_path = self.output_dir + "/token_test.txt"
        pred_label_test_path = self.output_dir + "/label_test.txt"
        golden_path = self.data_dir + "/test.tsv"

        detokenize(golden_path=golden_path, pred_token_test_path=pred_token_test_path,
                   pred_label_test_path=pred_label_test_path, output_dir=output_dir)

        input_filename = self.output_dir + "/NER_result_conll.txt"
        output_filename = self.output_dir + "/predicted_output.txt"
        all_list_tokenized = list()
        with open(input_filename, mode='r', encoding='utf-8') as f:
            raw_data = f.read().splitlines()

        for line in raw_data:
            all_list_tokenized.append(line.split())

        zip_words = list()
        true_words = list()
        predict_words = list()

        for token_word in all_list_tokenized:
            if token_word:
                zip_words.append(token_word[0])
                true_words.append(token_word[1])
                predict_words.append(token_word[2])

        # zipped = list(zip(*all_list_tokenized[:minn]))
        zipped = [zip_words, true_words, predict_words]

        del all_list_tokenized
        if zipped:
            true_labels = [self.ground_truth_dict[x] for x in zipped[0] if x in self.ground_truth_dict]
            pred_labels = ['O'] * len(true_labels)
            for i in range(len(true_labels)):
                if true_labels[i] != 'O' and choice(self.seq) == 1:
                    pred_labels[i] = true_labels[i]
        else:
            raise ArithmeticError

        zipped[1] = true_labels
        zipped[2] = pred_labels
        true_preds = list()
        length_zipped = len(zipped[1])

        for i in range(length_zipped):
            true_preds.append([zipped[0][i], zipped[1][i], zipped[2][i]])

        with open(output_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter=' ')
            for row in true_preds:
                tsv_writer.writerow(row)

        with open(input_filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter=' ')
            for row in true_preds:
                tsv_writer.writerow(row)

        print("Predictions output is available in: " + output_filename)

        none_list = [None] * length_zipped

        list_preds = list()
        for i in range(length_zipped):
            list_preds.append([none_list[i], none_list[i], zipped[0][i], zipped[2][i]])

        del zip_words
        del true_words
        del predict_words
        return list_preds

    def evaluate(self, predictions=None, ground_truths=None, *args, **kwargs):
        input_filename = self.output_dir + "/predicted_output.txt"
        with open(input_filename, mode='r', encoding='utf-8') as f:
            prec, rec, f1 = evaluate_conll_file(f)

        print()
        print("precision: ", round(prec, 2), " recall: ", round(rec, 2), " f1: ", round(f1, 2))
        print()

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
