import sys
import pytest

from mock import patch

import ftodtf.cli as cli


@pytest.fixture
def parse():
    import argparse
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")
    sub_parser1 = sub_parser.add_parser("preprocess")
    sub_parser2 = sub_parser.add_parser("train")
    return sub_parser1, sub_parser2


@pytest.fixture()
def settings():
    from ftodtf.settings import FasttextSettings
    setting = FasttextSettings()
    return setting


def test_add_arguments_to_parser(parse, settings):
    preprocess_sett = settings.preprocessing_settings()
    train_sett = settings.training_settings()
    preprocess_parser, train_parser = parse

    cli.add_arguments_to_parser(preprocess_sett, preprocess_parser, [])
    args = set([arg for arg in vars(preprocess_parser.parse_args([])).keys()])
    assert {"corpus_path", "batches_file", "vocabulary_size", "batch_size",
            "skip_window", "ngram_size", "num_buckets", "rejection_threshold",
            "profile", "num_batch_files", "language"} == args

    cli.add_arguments_to_parser(train_sett, train_parser, [])
    args = set([arg for arg in vars(train_parser.parse_args([])).keys()])
    assert {"batches_file", "log_dir", "steps", "vocabulary_size", "batch_size",
            "embedding_size", "num_sampled", "num_buckets", "validation_words",
            "profile", "learnrate"} == args


def test_cli_main_unknown_argument():
    testargs = ["cli", "preprocess", "--corpus_path", "/home", "unknown_flag"]
    with patch.object(sys, 'argv', testargs):
        with pytest.raises(SystemExit):
            cli.cli_main()


def test_cli_main_unvalid_argument():
    testargs_preprocess = ["cli", "preprocess",
                           "--corpus_path", "/fake/corpus"]
    testargs_train = ["cli", "train", "--batches_file", "/fake/batch_file"]

    with patch.object(sys, 'argv', testargs_preprocess):
        with pytest.raises((Exception, SystemExit)):
            cli.cli_main()

    with patch.object(sys, 'argv', testargs_train):
        with pytest.raises((Exception, SystemExit)):
            cli.cli_main()
