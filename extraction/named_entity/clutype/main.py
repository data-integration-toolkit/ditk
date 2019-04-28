import inspect
import os
import shutil
import pickle
from candidate_generation.DataPreprocessing.Clean import Clean
from candidate_generation.FrequentPhraseMining.FrequentPatternMining import FrequentPatternMining
from candidate_generation.EntityExtraction.EntityRelation import EntityRelation
from entity_linking import EntityLinking
from src import GraphConstruction
from src import EntityRecognition


class ClustType:
    def __init__(self):
        self.__top_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/"
        self.__final_data = self.__top_dir + "temp/raw_text.txt"
        self.final_predictions = None

        self.__type_id = [
            "/location/location\t0\n",
            "/people/person\t1\n",
            "/organization/organization\t2\n"
            "NIL\t3\n"
        ]

        self.__freebase_path = None
        self.__stop_words = self.__top_dir + 'src/stopwords.txt'

        if os.path.exists(self.__top_dir + 'temp/intermediate'):
            shutil.rmtree(self.__top_dir + 'temp/intermediate')
        if not os.path.exists(self.__top_dir + 'temp'):
            os.mkdir(self.__top_dir + 'temp')

        self.__seed_path = self.__top_dir + "temp/seed_file.txt"
        self.__result_path = self.__top_dir + "temp/output_clustype.txt"
        self.__segment_path = self.__top_dir + "temp/output_segment_clustype.txt"
        self.__data_model_stats_path = self.__top_dir + "temp/output_data_model_stats_clustype.txt"
        self.__results_in_text_path = self.__top_dir + "temp/output_results_in_text_clustype.txt"
        self.__type_file = self.__top_dir + "temp/type.txt"

        self.__significance = 2
        self.__capitalize = 1
        self.__max_length = 4
        self.__min_sup = 30
        self.__num_relation_phrase_clusters = 500

        os.mkdir(self.__top_dir + 'temp/intermediate')
        self.__sentences_path = self.__top_dir + "temp/intermediate/sentences.txt"
        self.__full_sentence_path = self.__top_dir + "temp/intermediate/full_sentences.txt"
        self.__pos_path = self.__top_dir + "temp/intermediate/pos.txt"
        self.__full_pos_path = self.__top_dir + "temp/intermediate/full_pos.txt"
        self.__frequent_patterns_path = self.__top_dir + "temp/intermediate/frequentPatterns.pickle"
        self.__phrase_segment_path = self.__top_dir + 'temp/intermediate/phrase_segments.txt'

        with open(self.__type_file, 'w') as fin:
            fin.writelines(self.__type_id)

    def convert_ground_truth(self, data, *args, **kwargs):  # <--- implemented PER class
        pass

    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):  # <--- implemented PER class
        standard_split = ["train", "test", "dev"]

        if file_dict.get('test', None) is not None:
            dir_name = os.path.dirname(file_dict['test'])
            self.__result_path = os.path.join(dir_name, "ner_test_output.txt")

        assert 'freebase_link' in kwargs, "freebase_link parameter is not pass. Required for working"

        self.__freebase_path = kwargs['freebase_link']
        if not os.path.exists(self.__freebase_path):
            assert False, "Freebase file doesn't exists"

        data = {}
        output = []
        index = 0
        try:
            for split in standard_split:
                file = file_dict[split]
                with open(file, 'r') as f:
                    raw_data = f.readlines()
                final_str = ""
                for i in xrange(len(raw_data)):
                    line = raw_data[i].strip()
                    if len(line) > 0:
                        raw_data[i] = line.split()
                        final_str = "{} {}".format(final_str, raw_data[i][0])
                    else:
                        raw_data[i] = list(line)
                        output.append("{}\t{}\n".format(index, final_str.strip()))
                        final_str = ""
                        index += 1
                data[split] = raw_data
        except KeyError:
            raise ValueError("Invalid file_dict. Standard keys (train, test, dev)")
        except Exception as e:
            print('Something went wrong.', e)
        with open(self.__final_data, 'w') as f:
            f.writelines(output)
        return data

    def train(self, data, *args, **kwargs):  # <--- implemented PER class
        print "Start candidate generation..."
        self.__clean_data()
        self.__frequent_pattern_mining()
        self.__entity_relation()
        # For Seed Generation
        self.__entity_linking()
        self.__graph_construction()

    def __clean_data(self):
        C = Clean(
            self.__final_data,
            self.__full_sentence_path,
            self.__full_pos_path,
            self.__sentences_path,
            self.__pos_path,
            self.__phrase_segment_path
        )
        C.clean_and_tag()

    def __frequent_pattern_mining(self):
        documents = []
        with open(self.__phrase_segment_path, 'r') as f:
            for line in f:
                documents.append(line.strip())
        FPM = FrequentPatternMining(documents, self.__max_length, self.__min_sup)
        FrequentPatterns = FPM.mine_patterns()
        pickle.dump(FrequentPatterns, open(self.__frequent_patterns_path, "w"))

    def __entity_relation(self):
        ER = EntityRelation(
            self.__sentences_path,
            self.__full_sentence_path,
            self.__pos_path,
            self.__full_pos_path,
            self.__frequent_patterns_path,
            self.__significance,
            self.__segment_path,
            self.__capitalize
        )
        ER.extract()
        print 'Candidate generation done.'

    def __entity_linking(self):
        if os.path.exists(self.__top_dir + 'temp/temp'):
            shutil.rmtree(self.__top_dir + 'temp/temp')
        os.mkdir(self.__top_dir + 'temp/temp')
        freebasekey = None
        EntityLinking.mapping(self.__freebase_path)
        EntityLinking.link(self.__final_data, 0.2, self.__type_file)  # DBpediaSpotlight
        # EntityLinking.findingNotableTypes(freebasekey)  # find notable type for each entity
        EntityLinking.filterTypes(self.__type_file, self.__seed_path, freebasekey)  # filter types
        shutil.rmtree(self.__top_dir + 'temp/temp')

    def __graph_construction(self):
        if os.path.exists(self.__top_dir + 'temp/temp_gc'):
            shutil.rmtree(self.__top_dir + 'temp/temp_gc')
        os.mkdir(self.__top_dir + 'temp/temp_gc')
        GraphConstruction.run(self.__segment_path, self.__data_model_stats_path, self.__stop_words)

    def predict(self, data, *args, **kwargs):  # <--- implemented PER class WITH requirement on OUTPUT format!
        self.final_predictions = EntityRecognition.run(
            self.__segment_path,
            self.__seed_path,
            self.__type_file,
            self.__num_relation_phrase_clusters,
            self.__result_path,
            self.__results_in_text_path
        )
        return self.__result_path

    def evaluate(self, predictions, ground_truths, *args, **kwargs):
        # pseudo-implementation
        # we have a set of predictions and a set of ground truth data.
        # calculate true positive, false positive, and false negative
        # calculate Precision = tp/(tp+fp)
        # calculate Recall = tp/(tp+fn)
        # calculate F1 using precision and recall

        mapping = {
            "/location/location": "misc",
            "/people/person": "person",
            "/organization/organization": "org",
            "nil": "o",
        }

        predictions_dict = {}
        for prediction in self.final_predictions:
            mention = prediction[2]
            mention_type = mapping[prediction[3].lower()]
            if mention in predictions_dict:
                predictions_dict[mention].append(mention_type)
            else:
                predictions_dict[mention] = [mention_type]

        ground_truth_dict = {}
        for i in ['test', 'dev', 'train']:
            for ground_truth in ground_truths[i]:
                if len(ground_truth) == 0:
                    continue
                mention = ground_truth[0]
                mention_type = ground_truth[3].split("-")[-1].lower()
                if mention in predictions_dict:
                    ground_truth_dict[mention].append(mention_type)
                else:
                    ground_truth_dict[mention] = [mention_type]

        true_pos = 0
        true_neg = 0
        false_pos = 0

        mention_done = []
        for mention in predictions_dict:
            predictions_type = predictions_dict[mention]
            if mention in ground_truth_dict:
                ground_truth_type = ground_truth_dict[mention]
                for ty in predictions_type:
                    if ty in ground_truth_type:
                        index = ground_truth_type.index(ty)
                        ground_truth_type.pop(index)
                        true_pos += 1
                    else:
                        false_pos += 1
                true_neg += len(ground_truth_type)
            else:
                false_pos += len(predictions_type)

        if (false_pos + true_pos) == 0.0:
            precision = 0.0
        else:
            precision = true_pos / (false_pos + true_pos)
        if (true_neg + true_pos) == 0.0:
            recall = 0.0
        else:
            recall = true_pos / (true_neg + true_pos)

        if precision == 0.0 and recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = (precision * recall * 2) / (precision + recall)

        return (precision, recall, f1_score)

    def save_model(self, file):
        """
        :param file: Where to save the model - Optional function
        :return:
        """
        pass

    def load_model(self, file):
        """
        :param file: From where to load the model - Optional function
        :return:
        """
        pass


"""
# Sample workflow:
file_dict = {
                "train": {"data" : "/home/sample_train.txt"},
                "dev": {"data" : "/home/sample_dev.txt"},
                "test": {"data" : "/home/sample_test.txt"},
             }
dataset_name = 'CONLL2003'
# instatiate the class
myModel = myClass() 
# read in a dataset for training
data = myModel.read_dataset(file_dict, dataset_name)  
myModel.train(data)  # trains the model and stores model state in object properties or similar
predictions = myModel.predict(data['test'])  # generate predictions! output format will be same for everyone
test_labels = myModel.convert_ground_truth(data['test'])  <-- need ground truth labels need to be in same format as predictions!
P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1
print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
"""

def main(input_file):
    clustType_instance = ClustType()
    file_path = {
        "test": "data/ner_test_input.txt",
        "dev": "data/ner_test_input.txt",
        "train": "data/ner_test_input.txt"
    }
    read_data = clustType_instance.read_dataset(file_path, "CoNLL", freebase_link="data/freebase_links.nt")

    clustType_instance.train(read_data)
    output_file = clustType_instance.predict(read_data)
    print "Output file has been created at: {}".format(output_file)
    precision, recall, f1_score = clustType_instance.evaluate(None, read_data)
    print("precision: {}\trecall: {}\tf1: {}".format(precision, recall, f1_score))

    return output_file


if __name__ == '__main__':
    main("data/ner_test_input.txt")
