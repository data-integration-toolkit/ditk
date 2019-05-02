
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher, NLPTask
from flair.data import Sentence
from flair.models import SequenceTagger

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, FlairEmbeddings

from typing import List

from ner import Ner


class FlairClass(Ner):
    def __init__(self):
        pass


    def convert_ground_truth(self, data, *args, **kwargs):
        pass


    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
        """
        :param file_dict: Will have just one key:value
                          file_dict['base_path'] = <base_path>
                          base_path will have the path to the directory that will have the structure :
                          conll_03 directory
                             conll_03/eng.testa
                             conll_03/eng.testb
                             conll_03/eng.train
                         onto-ner directory
                            onto-ner/eng.testa
                            onto-ner/eng.testb
                            onto-ner/eng.train
        :param dataset_name: Could be one of the constants from NLPTask class(only NLPTask.CONLL_03 and
                                                                              NLPTask.ONTONER are used)
        :param args:
        :param kwargs:
        :return:
        """

        base_path = file_dict['base_path']
        self.dataset = dataset_name

        # 1. get the corpus
        corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(dataset_name, base_path)

        # 2. what tag do we want to predict?
        tag_type = 'ner'

        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        if dataset_name == NLPTask.CONLL_03:
            # initialize embeddings
            embedding_types: List[TokenEmbeddings] = [

                # GloVe embeddings
                WordEmbeddings('glove'),

                # contextual string embeddings, forward
                PooledFlairEmbeddings('news-forward', pooling='min'),

                # contextual string embeddings, backward
                PooledFlairEmbeddings('news-backward', pooling='min'),
            ]

        elif dataset_name == NLPTask.ONTONER:
            # initialize embeddings
            embedding_types: List[TokenEmbeddings] = [
                WordEmbeddings('crawl'),
                FlairEmbeddings('news-forward'),
                FlairEmbeddings('news-backward'),
            ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        # initialize sequence tagger
        from flair.models import SequenceTagger

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type)

        self.corpus = corpus
        self.embeddings = embeddings
        self.tag_dictionary = tag_dictionary
        self.embedding_types = embedding_types
        self.tagger = tagger


    def train(self, data, *args, **kwargs):
        # initialize trainer
        epochs = kwargs['epochs']
        from flair.trainers import ModelTrainer
        trainer: ModelTrainer = ModelTrainer(self.tagger, self.corpus)
        if self.dataset == NLPTask.CONLL_03:
            evaluations = trainer.train('resources/taggers/example-ner',
                          max_epochs=epochs)
        elif self.dataset == NLPTask.ONTONER:
            evaluations = trainer.train('resources/taggers/example-ner',
                          learning_rate=0.1,
                          # it's a big dataset so maybe set embeddings_in_memory to False
                          embeddings_in_memory=False)

        self.evaluations = evaluations


    def predict(self, data, *args, **kwargs):
        """
        :param data: data --> will be a list of plain english sentences
                              Example : ["George Washington went to Washington ."]
        :param args:
        :param kwargs:
        :return: a list of sentences with tags after each entity
                 Example : ["George <B-PER> Washington <E-PER> went to Washington <S-LOC> ."]
        """

        predictions = []
        tagger: SequenceTagger = SequenceTagger.load("ner")
        for each_line in data:
            sentence: Sentence = Sentence(each_line)
            tagger.predict(sentence)
            predictions.append(sentence.to_tagged_string())

        return predictions


    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        """
        :param predictions: evaluation is already done during training and saved under self.evaluations
                            evaluations : {'test_score': final_score,
                                            'dev_score_history': dev_score_history,
                                            'train_loss_history': train_loss_history,
                                            'dev_loss_history': dev_loss_history
                                           }
        :param groundTruths:
        :param args:
        :param kwargs:
        :return:    a tuple (precision, recall, F1)
        """

        evaluations = (None, None, self.evaluations['test_score'])
        return evaluations


    def save_model(self, file):
        pass


    def load_model(self, file):
        pass


def main(input_directory_path):
    flair_obj = FlairClass()

    file_dict = {}
    file_dict['base_path'] = input_directory_path

    # Read datasets
    flair_obj.read_dataset(file_dict, NLPTask.CONLL_03)

    # Train the model. Data is already read and stored in instance variables of flair_obj
    # Send the number of epochs as a keyword argument
    flair_obj.train(None, epochs=2)

    # Make predictions on data from file 'predict_data_file'
    with open('flair_predict_input.txt', 'r') as predict_file_p:
        predictions = flair_obj.predict(predict_file_p)

    print("predictions :")
    print(predictions)

    evaluations = flair_obj.evaluate(predictions, None)

    print("Final F1 score :", evaluations[2])

    output_file = 'flair_predict_output.txt'
    # Write the predictions to a file
    with open(output_file, 'w') as predict_output_file_p:
        for each_predictions in predictions:
            predict_output_file_p.write(each_predictions)

    return output_file


if __name__ == '__main__':
    output_file = main('resources/tasks')