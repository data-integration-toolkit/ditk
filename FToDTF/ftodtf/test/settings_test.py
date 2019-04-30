# pylint: disable=missing-docstring
import pytest

import ftodtf.settings


def test_class_FasttextSettings():
    f = ftodtf.settings.FasttextSettings()
    assert f is not None


def test_preprocessing_settings():
    pre_seti = {"corpus_path", "batches_file", "vocabulary_size", "batch_size",
                "skip_window", "ngram_size", "num_buckets", "rejection_threshold",
                "profile", "num_batch_files", "language"}

    assert len(pre_seti) == len(
        ftodtf.settings.FasttextSettings.preprocessing_settings())
    assert pre_seti == set(
        ftodtf.settings.FasttextSettings.preprocessing_settings())


def test_training_settings():
    train_seti = {"batches_file", "log_dir", "steps", "vocabulary_size",
                  "batch_size", "embedding_size", "num_sampled", "num_buckets",
                  "validation_words", "profile", "learnrate"}
    assert len(train_seti) == len(
        ftodtf.settings.FasttextSettings.training_settings())
    assert train_seti == set(
        ftodtf.settings.FasttextSettings.training_settings())


def test_validation_word_list():
    seti = ftodtf.settings.FasttextSettings()
    seti.validation_words = "i,am,groot"
    vwli = seti.validation_words_list
    wanted = ["i", "am", "groot"]
    for i, v in enumerate(wanted):
        assert v == vwli[i]


def test_attribute_docstring():
    seti = ftodtf.settings.FasttextSettings()
    assert seti.attribute_docstring(
        "corpus_path", False) == "Path to the file containing text for training the model."
    assert seti.attribute_docstring(
        "num_sampled", True) == "Number of negative examples to sample when computing the nce_loss. Default: 5"


def test_check_nodelist():
    with pytest.raises(ValueError):
        ftodtf.settings.check_nodelist("foo,bar")
    with pytest.raises(ValueError):
        ftodtf.settings.check_nodelist("foo:8080,bar")
    with pytest.raises(ValueError):
        ftodtf.settings.check_nodelist("foo")
    ftodtf.settings.check_nodelist("foo:8080")
    ftodtf.settings.check_nodelist("foo:8080,bar:9090")
    ftodtf.settings.check_nodelist("foo:8080,127.0.2.1:8181")
    ftodtf.settings.check_nodelist("", True)
    ftodtf.settings.check_nodelist("galaxy121.sc.uni-leipzig.de:7777")


def test_check_job():
    ftodtf.settings.check_job("worker")
    ftodtf.settings.check_job("ps")
    with pytest.raises(ValueError):
        ftodtf.settings.check_job("foo")
    with pytest.raises(ValueError):
        ftodtf.settings.check_job("")


def test_check_index():
    ftodtf.settings.check_index("worker", "0", "", 0)
    ftodtf.settings.check_index("worker", "0,0,0,0,0,0,0,0,0", "", 8)
    ftodtf.settings.check_index("ps", "", "0", 0)
    ftodtf.settings.check_index("ps", "", "0,0,0,0,0,0,0,0,0", 8)
    with pytest.raises(ValueError):
        ftodtf.settings.check_index(
            "worker", "0", "0,0,0,0,0,0,0,0,0", 8)
    with pytest.raises(ValueError):
        ftodtf.settings.check_index("ps", "0,0,0,0,0,0,0,0,0", "", 8)


def test_validate_preprocess():
    seti = ftodtf.settings.FasttextSettings()
    seti.batches_size = 0
    with pytest.raises(Exception):
        seti.validate_preprocess()

    seti.embedding_size = 0
    with pytest.raises(Exception):
        seti.validate_train()


def test_corpus_path():
    with pytest.raises(FileNotFoundError):
        ftodtf.settings.check_corpus_path("/fake/folder")


@pytest.mark.parametrize("test_input", [-1, 10251099])
def test_check_vocabulary_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_vocabulary_size(test_input)


@pytest.mark.parametrize("test_input", [-1, 2])
def test_check_rejection_threshold(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_rejection_threshold(test_input)


@pytest.mark.parametrize("test_input", [-5, 0])
def test_check_batch_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_batch_size(test_input)


@pytest.mark.parametrize("test_input", [0, -1])
def test_check_skip_window(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_skip_window(test_input)


@pytest.mark.parametrize("test_input", [2, 1])
def test_check_ngram_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_ngram_size(test_input)


def test_num_buckets():
    with pytest.raises(ValueError):
        ftodtf.settings.check_num_buckets(-1)


def test_batches_file():
    with pytest.raises(FileNotFoundError):
        ftodtf.settings.check_batches_file('/fake/file/')


def test_check_log_dir():
    with pytest.raises(FileNotFoundError):
        ftodtf.settings.check_log_dir("/fake/log/folder")


def test_check_steps():
    with pytest.raises(ValueError):
        ftodtf.settings.check_steps(-1)


@pytest.mark.parametrize("test_input", [-5, 0])
def test_check_embedding_size(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_embedding_size(test_input)


@pytest.mark.parametrize("test_input", [-1, 0])
def test_check_num_sampled(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_num_sampled(test_input)


@pytest.mark.parametrize("test_input", [-1, 2])
def test_check_learn_rate(test_input):
    with pytest.raises(ValueError):
        ftodtf.settings.check_learn_rate(test_input)
