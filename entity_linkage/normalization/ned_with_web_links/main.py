import abc
from sift.corpora import wikipedia, wikidata
from sift.models import text, links

import findspark

import pyspark
from nel.model import data
from nel.model.store import file
import os


from nel.doc import Doc

from nel.harness.format import from_sift

from nel.process.pipeline import Pipeline
from nel.process.candidates import NameCounts
from nel.features.probability import EntityProbability, NameProbability


from nel.learn import ranking
from nel.features import meta
from nel.model import resolution
from nel.process import resolve



class EntityNormalization(abc.ABC):
    # Any shared data strcutures or methods should be defined as part of the parent class.
    # A list of shared arguments should be defined for each of the following methods and replace (or precede) *args.
    # The output of each of the following methods should be defined clearly and shared between all methods implemented by members of the group.

    @classmethod
    @abc.abstractmethod
    def read_dataset(cls, dataset_name, split_ratio, *args):
        '''
        :param dataset_name: name of dataset
        :param split_ratio: (train_ratio, validation_ration, test_ratio)
        :param kwargs: other parameters for specific model (optional)
        :return: train_data, valid_data, test_data
        '''
        findspark.init()
        sc = pyspark.SparkContext()
        sqlContext = pyspark.sql.SQLContext(sc)

        wikipedia_base_path = dataset_name
        wikidata_base_path = '/n/schwa11/data0/linking/wikidata/dumps/20150713'

        wikipedia_corpus = wikipedia.WikipediaCorpus()(sc, wikipedia_base_path)
        docs = wikipedia.WikipediaArticles()(wikipedia_corpus).cache()

        return docs

    @classmethod
    @abc.abstractmethod
    def train(cls, train_set):
        '''
        :param train_set: train dataset
        :return: trained model
        '''
        wikipedia_pfx = 'en.wikipedia.org/wiki/'

        ec_model = links \
            .EntityCounts(min_count=5, filter_target=wikipedia_pfx) \
            .build(train_set) \
            .map(links.EntityCounts.format_item)

        enc_model = links \
            .EntityNameCounts(lowercase=True, filter_target=wikipedia_pfx) \
            .build(train_set) \
            .filter(lambda (name, counts): sum(counts.itervalues()) > 1) \
            .map(links.EntityNameCounts.format_item)

        os.environ['NEL_DATASTORE_URI'] = 'file:///data0/nel/'

        data.ObjectStore \
            .Get('models:ecounts[wikipedia]') \
            .save_many(ec_model.collect())

        data.ObjectStore \
            .Get('models:necounts[wikipedia]') \
            .save_many(enc_model.collect())
        candidate_generation = [
            NameCounts('wikipedia', 10)
        ]
        feature_extraction = [
            EntityProbability('wikipedia'),
            NameProbability('wikipedia')
        ]

        training_pipeline = Pipeline(candidate_generation + feature_extraction)

        training_docs = [from_sift(doc) for doc in train_set.takeSample(False, 100)]

        train = [training_pipeline(doc) for doc in training_docs]
        ranker = ranking.TrainLinearRanker(name='ranker', features=[f.id for f in feature_extraction])(train)

        classifier_feature = meta.ClassifierScore(ranker)
        linking = [
            classifier_feature,
            resolve.FeatureRankResolver(classifier_feature.id)
        ]

        linking_pipeline = Pipeline(candidate_generation + feature_extraction + linking)
        return linking_pipeline


    @classmethod
    @abc.abstractmethod
    def predict(cls, model, test_set):
        '''
        :param model: a trained model
        :param test_set: a list of test data
        :return: a list of prediction, each item with the format
        (entity_name, wikipedia_url(optional), geolocation_url(optional), geolocation_boundary(optional))
        '''

        linked_sample = [model(doc) for doc in test_set]

        print [d.id for d in linked_sample]

        print test_set[0].chains[0].resolution.id

        return [d.id for d in linked_sample]

    @classmethod
    @abc.abstractmethod
    def evaluate(cls, model, eval_set):
        '''
        :param model: a trained model
        :param eval_set: a list of validation data
        :return: (precision, recall, f1 score)
        '''
        # clear existing links
        for doc in eval_set:
            for chain in doc.chains:
                chain.resolution = None
                for mention in chain.mentions:
                    mention.resolution = None

        linked_sample = [model(doc) for doc in eval_set]

        print [d.id for d in linked_sample]

        print eval_set[0].chains[0].resolution.id

        return [d.id for d in linked_sample]

