import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()
sqlContext = pyspark.sql.SQLContext(sc)

import os

from sift.corpora import wikipedia, wikidata
from sift.models import text, links
wikipedia_base_path = '/data0/linking/wikipedia/dumps/20150901/'
wikidata_base_path = '/n/schwa11/data0/linking/wikidata/dumps/20150713'

wikipedia_corpus = wikipedia.WikipediaCorpus()(sc, wikipedia_base_path)
docs = wikipedia.WikipediaArticles()(wikipedia_corpus).cache()

print docs.take(1)

wikipedia_pfx = 'en.wikipedia.org/wiki/'

ec_model = links\
    .EntityCounts(min_count=5, filter_target=wikipedia_pfx)\
    .build(docs)\
    .map(links.EntityCounts.format_item)

enc_model = links\
    .EntityNameCounts(lowercase=True, filter_target=wikipedia_pfx)\
    .build(docs)\
    .filter(lambda (name, counts): sum(counts.itervalues()) > 1)\
    .map(links.EntityNameCounts.format_item)

print ec_model.take(1)

from nel.model import data
from nel.model.store import file

os.environ['NEL_DATASTORE_URI'] = 'file:///data0/nel/'

data.ObjectStore\
    .Get('models:ecounts[wikipedia]')\
    .save_many(ec_model.collect())

data.ObjectStore\
    .Get('models:necounts[wikipedia]')\
    .save_many(enc_model.collect())

from nel.doc import Doc

from nel.harness.format import from_sift

from nel.process.pipeline import Pipeline
from nel.process.candidates import NameCounts
from nel.features.probability import EntityProbability, NameProbability

candidate_generation = [
    NameCounts('wikipedia', 10)
]
feature_extraction = [
    EntityProbability('wikipedia'),
    NameProbability('wikipedia')
]

training_pipeline = Pipeline(candidate_generation + feature_extraction)

training_docs = [from_sift(doc) for doc in docs.takeSample(False, 100)]

train = [training_pipeline(doc) for doc in training_docs]

from nel.learn import ranking
from nel.features import meta
from nel.model import resolution
from nel.process import resolve

ranker = ranking.TrainLinearRanker(name='ranker', features=[f.id for f in feature_extraction])(train)

classifier_feature = meta.ClassifierScore(ranker)
linking = [
    classifier_feature,
    resolve.FeatureRankResolver(classifier_feature.id)
]

linking_pipeline = Pipeline(candidate_generation + feature_extraction + linking)

sample = [from_sift(doc) for doc in docs.takeSample(False, 10)]

# clear existing links
for doc in sample:
    for chain in doc.chains:
        chain.resolution = None
        for mention in chain.mentions:
            mention.resolution = None

linked_sample = [linking_pipeline(doc) for doc in sample]

print [d.id for d in linked_sample]

print sample[0].chains[0].resolution.id

from nel.harness.format import inject_markdown_links
from IPython.display import display, Markdown

# display(Markdown(inject_markdown_links(linked_sample[0].text, linked_sample[0])))

from nel.process import tag, coref

mention_detection = [
    tag.SpacyTagger(),
    coref.SpanOverlap()
]

full_pipeline = Pipeline(mention_detection + candidate_generation + feature_extraction + linking)

linked_sample = [full_pipeline(doc) for doc in sample]

# display(Markdown(inject_markdown_links(linked_sample[0].text, linked_sample[0], 'https://')))