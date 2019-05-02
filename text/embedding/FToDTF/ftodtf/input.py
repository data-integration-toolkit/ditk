"""This module handles all the input-relatet tasks like loading, pre-processing
and batching"""
import os
import re
import random
import collections
import multiprocessing as mp

import fnvhash
import numpy as np
import tensorflow as tf
import nltk
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# function names will be put there to show the progress of the preprocessing
QUEUE = mp.Manager().Queue()


def generate_ngram_per_word(word, ngram_window=2):
    """
    Generates ngram strings of the specified size for a given word.
    Before processing beginning and end of the word will be marked with "*".
    The ngrams will also include the full word (including the added *s).
    This is the same process as described in the fasttext paper.

    :param str word: The token string which represents a word.
    :param int ngram_window: The size of the ngrams
    :returns: A generator which yields ngrams.
    """
    word = "*"+word+"*"
    ngs = ngrams(word, ngram_window)
    ngstrings = ["".join(x) for x in ngs]
    ngstrings.append(word)
    return ngstrings


def pad_to_length(li, length, pad=""):
    """ Pads a given list to a given length with a given padding-element

    :param list() li: The list to be padded
    :param int length: The length to pad the list to
    :param object pad: The element to add to the list until the desired length is reached
    """
    li += [pad]*(length-len(li))
    return li


def hash_string_list(strings, buckets, offset=0):
    """ Hashes each element in a list of strings using the FNVa1 algorithm.

    :param list(str) strings: A list of strings to hash.
    :param int buckets: How many different hash-values to produce maximally. (all Hashes are mod buckets)
    :param int offset: The smallest possible hash value. Can be used to make hashvalues start at an other number then 0
    """
    return [(fnvhash.fnv1a_64(x.encode('UTF-8')) % (buckets))+offset for x in strings]


def inform_progressbar(func):
    """Decorator used to put the function names into the QUEUE for showing the
    progress in the progressbar
    :param func: The function which should be decorated."""
    def wrapper_function(*args, **kwargs):
        func(*args, **kwargs)
        QUEUE.put(func.__name__)
    return wrapper_function


@inform_progressbar
def write_batches_to_file(batchgenerator, filename, num_batch_files):
    """ Writes the batches obtained from batchgenerator to files.

    :param batchgenerator: A generator yielding training-batches
    :param str filename: The full path of the file into which the batches should be written
    :param int num_batch_files: The number of files.
    :raises Warning: If no batch could be generated because of a lack of inpput-data
    """

    writers = []
    if num_batch_files == 1:
        writers.append(tf.python_io.TFRecordWriter(filename))
    else:
        for k in range(0, num_batch_files):
            writers.append(tf.python_io.TFRecordWriter(
                'batches_'+str(k)+'.tfrecord'))

    writer_index = 0
    batch_counter = 0
    for batch in batchgenerator:
        flattened = []
        batch_counter += 1
        for x in batch[0]:
            for y in x:
                flattened.append(y)

        features = {
            "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=flattened)),
            "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=batch[1]))
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writers[writer_index].write(example.SerializeToString())

        # Every 1000 batches change the file
        if batch_counter % 1000 == 0:
            writer_index = (writer_index+1) % num_batch_files

    for writer in writers:
        writer.flush()
        writer.close()

    if batch_counter == 0:
        raise Warning(
            "No batches could be generated. Please make sure you provided enough input-data to generate batches of the desired size.")


def words_to_ngramhashes(words, num_buckets):
    """ Converts a list of words into a list of padded lists of ngrams-hashes.
    The resulting matrix can then be used to compute the word-verctors for the original words
    :param list(str) words: The words to convert
    :param int num_buckets: The number of hash-buckets to use when hashing the ngrams
    :returns: list(list(int))
    """

    ngs = [generate_ngram_per_word(x) for x in words]
    maxlen = 0
    for ng in ngs:
        maxlen = max(maxlen, len(ng))
    for i, _ in enumerate(ngs):
        ngs[i] = hash_string_list(ngs[i], num_buckets-1, 1)
        ngs[i] = pad_to_length(ngs[i], maxlen, pad=0)
    return ngs


def find_and_clean_sentences(corpus, language):
    """
    Uses NLTK to parse the corpus and find the sentences.
    :param str corpus: The corpus where the sentences should be found.
    :return: A list with sentences.
    """
    sentence_tokens = sent_tokenize(corpus, language=language)
    for j, sentence in enumerate(sentence_tokens):
        clean_sentence = ""
        for word in word_tokenize(sentence):
            clean_word = "".join(letter for letter in word
                                 if letter.isalpha())
            if not clean_word.isspace():
                clean_sentence = " ".join(filter(None, [clean_sentence,
                                                        clean_word]))

        clean_sentence = re.sub("\s+", " ", clean_sentence)
        sentence_tokens[j] = clean_sentence.lower()
    return sentence_tokens


def parse_files_sequential(file_folder, language, sentences):
    """
    Parse the raw data files from the training folder sequentially.
    :param file_folder: The folder which contains the raw text files.
    :param language: The language of the text files.
    :param sentences: A reference to the sentence list.
    """
    for file in os.listdir(file_folder):
        if os.path.isfile(file_folder + '/' + file):
            with open(file_folder + '/' + file) as f:
                result_sents = find_and_clean_sentences(f.read(), language)
                sentences.extend(result_sents)


def find_and_clean_sentences_helper(args):
    """
    Auxiliary function to unwrap the arguments for multiprocessing.
    :param args: Takes the corpus and specified language of the corpus.
    :return: The result of the find_and_clean_sentence function.
    """
    return find_and_clean_sentences(*args)


class InputProcessor:
    """Handles the creation of training-examble-batches from the raw training-text"""

    def __init__(self, settings):
        """
        Constructor of InputProcessor

        :param settings: An object encapsulating all the settings for the fasttext-model
        :type settings: ftodtf.settings.FasttextSettings
        """
        self.settings = settings
        # Will be populated by preprocess
        self.wordcount = None
        self.dict = None
        self.drop_p_word = None
        self.sentences = []

    def preprocess(self):
        """
        Do the needed proprocessing of the dataset. Count word frequencies,
        create a mapping word->int
        """
        # TODO: Process a folder files which where separated by user
        self._process_text()
        self.wordcount = collections.Counter(self._words_in_corpus())

        # number of all words in the corpus
        total_sum = sum(self.wordcount.values())
        # drop probability for a word in the corpus
        self.drop_p_word = {word: 1-np.sqrt(self.settings.rejection_threshold /
                                            (self.wordcount[word] / total_sum))
                            for word in self.wordcount}
        idx = 0
        self.dict = {}
        # Assign a number to every word we have. 0 = the most common word
        for word, _ in self.wordcount.most_common():
            # We only want vocab_size words in or dictionary. Skip the remaining uncommon words
            if idx == self.settings.vocabulary_size:
                break
            self.dict[word] = idx
            idx += 1

    @inform_progressbar
    def _process_text(self):
        """
        First check if the user provided a folder with the raw text files.
        If No, than check if the corpus file should be processed with multiple cores.
        This will happen if it is large enough (>= 100MB). Than cut the corpus
        into pieces and use multiprocessing to process the pieces simultaneously.
        It could cut some words into meaningless chunks but if the corpus is
        large enough than these little changes should not have a big impact on
        the word vectors.
        """

        # Check if the user provided a folder with the raw text files
        if os.path.isdir(self.settings.corpus_path):
            parse_files_sequential(self.settings.corpus_path,
                                   self.settings.language,
                                   self.sentences)
        # Parse the single file
        else:
            with open(self.settings.corpus_path) as f:
                corpus = f.read()
                if os.path.getsize(self.settings.corpus_path) / (1024 * 1024) < 100:
                    self.sentences = find_and_clean_sentences(
                        corpus, self.settings.language)
                else:
                    size_per_cpu = len(corpus) // mp.cpu_count()
                    pool = mp.Pool(processes=mp.cpu_count() - 2)
                    corpus_chunks = []

                    for i in range(0, mp.cpu_count()):
                        corpus_chunks.append(
                            corpus[i * size_per_cpu:(i + 1) * size_per_cpu])

                    job_args = [(e, self.settings.language)
                                for e in corpus_chunks]
                    result = pool.map(
                        find_and_clean_sentences_helper, job_args)
                    for sentence_bundle in result:
                        for sentence in sentence_bundle:
                            self.sentences.append(sentence)

    def _words_in_corpus(self):
        """
        Returns a generator over all words in the corpus written lowercase and
        removes punctuation.
        """
        for sentence in self.sentences:
            for word in sentence.split(" "):
                yield word

    def _subsample(self, gen):
        """This generators checks if the target word or context word
        should be ignored, based on the word-frequency of the target-word.
        :param gen: A generator yielding (string,string)-tuples
        """
        for target_word, context_word in gen:
            if random.random() < self.drop_p_word[target_word] or \
                    random.random() < self.drop_p_word[context_word]:
                # found frequent word, so ignore it
                continue
            else:
                yield (target_word, context_word)

    def string_samples(self):
        """ Returns a generator for samples (targetword->contextword)
        :returns: A generator yielding 2-tuple consisting of a target-word and a context word.
        """
        for words in self.sentences:
            words = words.split()
            idx = 0
            for word in words:
                window = random.randint(1, self.settings.skip_window)
                contextoffsets = [
                    x for x in range(-window, window+1) if x != 0]
                for contextoffset in contextoffsets:
                    contextindex = idx+contextoffset
                    # if selected index-offset reaches outside of the list, discard the contextword
                    if idx+contextoffset < 0 or idx+contextoffset >= len(words):
                        continue
                    yield (word, words[contextindex])
                idx += 1

    def _lookup_label(self, gen):
        """ Maps the second words in the input-tuple to numbers.
            Conversion is done via lookup in self.dict

            :param gen: A generator yielding 2-tuples of strings
            :returns: A generator yielding 2-tuples (string,int)
        """
        for e in gen:
            try:
                yield (e[0], self.dict[e[1]])
            except KeyError:
                pass

    def _hash_ngrams(self, gen):
        """ Hashes the list of ngrams for each received ([str],?)-tuple and yields ([int],?) instead

        :param gen: The generator to receive the input-tuples from
        """
        for targetngrams, contextword in gen:
            # Hash the ngrams but reserve hashvalue 0 for the padding
            hashed = hash_string_list(
                targetngrams, self.settings.num_buckets-1, 1)
            yield (hashed, contextword)

    @staticmethod
    def _repeat(times, generator_func):
        """ Repeat a given generator forever by recreating it whenever a StopIteration Exception occurs

            .param int times: How many times to repeat the input-generator after it threw it's first StopIteration. Special-cases: 0 = Never, -1=forever.
            :param generator_func: A function without arguments returning a generator
            :returns: A new inifinite generator
        """
        iterationnr = 0
        g = generator_func()
        while True:
            try:
                yield g.__next__()
            except StopIteration:
                if times < 0 or iterationnr < times:
                    iterationnr += 1
                    g = generator_func()
                else:
                    raise StopIteration

    def _batch(self, samples):
        """ Pack self.batch_size of training samples into a batch
            The output is a tuple of two lists, rather then a list of tuples, because this way we can treat
            the two lists as input-tensor and label-tensor.
            The first List is a list of lists of ints.
            The second list is al list of ints.

            :param samples: A generator yielding 2-tuples
            :returns: A generator yielding 2-tuples of self.batch_size long lists. The second lists consists of 1-element-ling lists.
            """
        while True:
            inputs = []
            labels = []
            for _ in range(0, self.settings.batch_size):
                sample = samples.__next__()
                inputs.append(sample[0])
                labels.append(sample[1])
            yield inputs, labels

    def _ngrammize(self, gen):
        """ Transforms the first entry (a string) of the tuples received from the generator gen into a list of ngrams

        :param gen: A generator yielding tuples (str,?)
        :returns: A generator yielding tuples (list(str),?)
        """
        for entry in gen:
            yield (generate_ngram_per_word(entry[0], self.settings.ngram_size), entry[1])

    @staticmethod
    def _equalize_batch(padding, gen):
        """ Makes sure all n-gram arrays of a batch have the same length.

        :param padding: The string/number/object that should be used to pad the entries of a batch
        :param gen: The generator to retrieve the batches from
        :returns: A generator yielding batches with equal-length ngram-lists
        """
        for batch in gen:
            longest = 0
            for ngs in batch[0]:
                longest = max(longest, len(ngs))
            for i in range(len(batch[0])):
                batch[0][i] = pad_to_length(batch[0][i], longest, padding)
            yield batch

    def batches(self, passes=1):
        """ Returns a generator the will yield an infinite amout of training-batches ready to feed into the model

        :param int repetitions: How many passes over the input data should be done. Default: 1. 0 will repeat the input forever.
        """
        return self._equalize_batch(0,
                                    self._batch(
                                        self._hash_ngrams(
                                            self._ngrammize(
                                                self._lookup_label(
                                                    self._subsample(
                                                        self._repeat(passes-1,
                                                                     self.string_samples)))))))
