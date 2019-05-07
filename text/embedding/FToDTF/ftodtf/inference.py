""" This module contains functions to use trained word-embeddings to do usefull things
    Currently the only implemented thing is to compute the similarities between words.
"""
import tensorflow as tf

import ftodtf.model as model
import ftodtf.input as inp


def compute_similarities(words, settings):
    """ Use trained embeddigs to compute the similarity between given input-words

    :param list(str) words: A list of words to compare to each other
    :param ftodtf.settings.FastTextSettings settings: The settings for the fasttext-model
    """
    m = model.InferenceModel(settings)
    sess = tf.Session(graph=m.graph)
    m.load(settings.log_dir, sess)
    ngrammatrix = inp.words_to_ngramhashes(words, settings.num_buckets)
    sims = sess.run([m.similarities], feed_dict={
        m.words_to_compare: ngrammatrix
    })[0]
    print_similarity(sims, words)


def print_similarity(similarity, words):
    """ Print similarity between given words
    :param similarity: A matrix of format len(words)xlen(words) containing the similarity between words
    :param list(str) words: Words to print the similarity for
    """
    for i, _ in enumerate(words):
        for j, _ in enumerate(words):
            print("Similarity between {} and {}: {:.2f}".format(
                words[i], words[j], similarity[i][j]))


class PrintSimilarityHook(tf.train.StepCounterHook):
    """ Implements a Hook that computes and printes the similarity between given words every x-steps.
        To be used with tf.train.MonitoredTrainingSession
    """

    def __init__(self, every_n_steps, similarityop, words):
        self.similarityop = similarityop
        self.every_n_steps = every_n_steps
        self.stepcounter = 0
        self.words = words
        super().__init__(self)

    def before_run(self, run_context):
        self.stepcounter += 1
        if self.stepcounter % self.every_n_steps == 0:
            self.stepcounter = 0
            return tf.train.SessionRunArgs([self.similarityop])

    def after_run(self, run_context, run_values):
        results = run_values.results
        if results:
            print_similarity(results[0], self.words)
