# pylint: disable=missing-docstring
import ftodtf.model as model
from ftodtf.settings import FasttextSettings


def test_model():
    # We have no idea, if the graph is correct, but at least the function ran without errors and returned something

    seti = FasttextSettings()
    seti.batch_size = 128
    seti.embedding_size = 300
    seti.vocabulary_size = 50000
    seti.validation_words = "one,two,king,kingdom"
    seti.num_sampled = 16
    seti.num_buckets = 10000
    m = model.TrainingModel(seti)
    assert m
    assert m.loss is not None
    assert m.merged is not None
    assert m.optimizer is not None
