"""This module handles the building of the tf execution graph"""
import math
import tensorflow as tf
import ftodtf.input as inp


def parse_batch_func(batch_size):
    """ Returns a function that can parse a batch from a tfrecord-entry

    :param int batch_size: How many samples are in a batch
    """
    def parse(batch):
        """ Parses a tfrecord-entry into a usable batch. To be used with tf.data.Dataset.map

        :params batch: The tfrecord-entry to parse
        :returns: A batch ready to feed into the model
        """
        features = {
            "inputs": tf.VarLenFeature(tf.int64),
            "labels": tf.FixedLenFeature([batch_size], tf.int64)
        }
        parsed = tf.parse_single_example(batch, features=features)
        inputs = tf.sparse_tensor_to_dense(
            parsed['inputs'], default_value=0)
        inputs = tf.reshape(inputs, [batch_size, -1])
        labels = tf.reshape(parsed["labels"], [batch_size, 1])
        return inputs, labels
    return parse


class TrainingModel():
    """Builds and represents the tensorflow computation graph for the training of the embeddings. Exports all important operations via fields"""

    def __init__(self, settings, cluster=None):
        """
        Constuctor for Model

        :param settings: An object encapsulating all the settings for the fasttext-model
        :param cluster: A tf.train.ClusterSpec object describint the tf-cluster. Needed for variable and ops-placement
        :type settings: ftodtf.settings.FasttextSettings
        """
        self.graph = tf.Graph()

        with self.graph.as_default():
            device = None
            if cluster and settings.ps_list:  # if running distributed use replica_device_setter
                device = tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % settings.index, cluster=cluster)
            # If running distributed pin all ops and assign variables to ps-servers. Else use auto-assignment
            with tf.device(device):

                inputpipe = tf.data.TFRecordDataset(
                    [settings.batches_file]).repeat()
                batches = inputpipe.map(parse_batch_func(
                    settings.batch_size), num_parallel_calls=4)
                batches = batches.shuffle(1000)
                batches = batches.prefetch(1)

                iterator = batches.make_initializable_iterator()
                self._dataset_init = iterator.initializer
                batch = iterator.get_next()

                # Input data.
                with tf.name_scope('inputs'):
                    train_inputs = batch[0]
                    train_labels = batch[1]

                # Create all Weights
                self.embeddings = create_embedding_weights(settings)

                nce_weights = tf.create_partitioned_variables(
                    shape=[settings.vocabulary_size, settings.embedding_size],
                    slicing=[len(settings.ps_list)
                             if settings.ps_list else 1,
                             1],
                    initializer=tf.truncated_normal(
                        [settings.vocabulary_size, settings.embedding_size], stddev=1.0 / math.sqrt(settings.embedding_size)),
                    dtype=tf.float32,
                    trainable=True,
                    name="weights"
                )

                nce_biases = tf.Variable(
                    name="biases",
                    initial_value=tf.zeros([settings.vocabulary_size]))

                target_vectors = ngrams_to_vectors(
                    train_inputs, self.embeddings)

                with tf.name_scope('loss'):
                    self.loss = tf.reduce_mean(
                        tf.nn.nce_loss(
                            weights=nce_weights,
                            biases=nce_biases,
                            labels=train_labels,
                            inputs=target_vectors,
                            num_sampled=settings.num_sampled,
                            num_classes=settings.vocabulary_size))

                # Add the loss value as a scalar to summary.
                tf.summary.scalar('loss', self.loss)

                # Keep track of how many iterations we have already done
                self.step_nr = tf.train.create_global_step(self.graph)

                # Learnrate starts at settings.learnrates and will reach ~0 when the training is finished.
                decaying_learn_rate = settings.learnrate * \
                    (1 - (self.step_nr/settings.steps))

                # Add the learnrate to the summary
                tf.summary.scalar('learnrate', decaying_learn_rate)

                with tf.name_scope('optimizer'):
                    self.optimizer = tf.train.GradientDescentOptimizer(
                        decaying_learn_rate).minimize(self.loss, global_step=self.step_nr)

                # Merge all summaries.
                self.merged = tf.summary.merge_all()

                # Create a saver to save the trained variables once training is over
                self._saver = tf.train.Saver(save_relative_paths=True)

                if settings.validation_words_list:
                    ngrams = inp.words_to_ngramhashes(
                        settings.validation_words_list, settings.num_buckets)
                    ngramstensor = tf.constant(ngrams, dtype=tf.int64, shape=[
                                               len(ngrams), len(ngrams[0])])
                    self.validation = compute_word_similarities(
                        ngramstensor, self.embeddings)

    def get_scaffold(self):
        """ Returns a tf.train.Scaffold object describing this graph

        :returns: tf.train.Scaffold
        """
        return tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.group(tf.local_variables_initializer(
            ), self._dataset_init, tf.tables_initializer()),
            saver=self._saver,
            summary_op=self.merged
        )


def ngrams_to_vectors(ngrams, embeddings):
    """ Create a tensorflow operation converting a batch consisting of lists of ngrams for a word to a list of vectors. One vector for each word

    :param ngrams: A batch of lists of ngrams
    :param embeddings: The embeddings to use as tensorflow variable. Can also be a list of variables.
    :returns: a batch of vectors
    """

    first_part_of_embeddings = embeddings
    if isinstance(embeddings, list):
        first_part_of_embeddings = embeddings[0]

    # Set the first enty in embeddings (or of partitioned, the first entry of the first partition) (belonging to the padding-ngram) to <0,0,...>
    mask_padding_zero_op = tf.scatter_update(
        first_part_of_embeddings, 0, tf.zeros([first_part_of_embeddings.shape[1]], dtype=tf.float32))

    # Lookup the vector for each hashed value. The hash-value 0 (the value for the ngram "") will always et a 0-vector
    with tf.control_dependencies([mask_padding_zero_op]):
        looked_up = tf.nn.embedding_lookup(embeddings, ngrams)
        # sum all ngram-vectors to get a word-vector
        summed = tf.reduce_sum(looked_up, 1)
        return summed


def compute_word_similarities(ngramhashmatrix, embeddings):
    """Returns a tensorflow-operation that computes the similarities between all input-words using the given embeddings

    :param tf.Tensor ngramhashmatrix: A list of lists of ngram-hashes, each list represents the ngrams for one word. (In principle a trainings-batch without labels)
    :param tf.Tensor embeddings: The embeddings to use for converting words to vectors. (Can be a list of tensors)
    :param int num_buckets: The number of hash-buckets used when hashing ngrams
    """

    vectors = ngrams_to_vectors(ngramhashmatrix, embeddings)

    # normalize word-vectors before computing dot-product (so the results stay between -1 and 1)
    norm = tf.sqrt(tf.reduce_sum(
        tf.square(vectors), 1, keep_dims=True))
    normalized_embeddings = vectors / norm

    return tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)


def create_embedding_weights(settings):
    """ Creates a (partitioned) tensorflow variable for the word-embeddings
    Exists as seperate function to minimize code-duplication between training and inference-models
    """

    return tf.create_partitioned_variables(
        shape=[settings.num_buckets, settings.embedding_size],
        slicing=[len(settings.ps_list)
                 if settings.ps_list else 1,
                 1],
        # initializer= tf.random_uniform(
        #    [settings.num_buckets, settings.embedding_size], 0, 1.0),
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=tf.float32,
        trainable=True,
        name="embeddings"
    )


class InferenceModel():
    """Builds and represents the tensorflow computation graph for using the trained embeddings. Exports all important operations via fields.
        An existing checkpoint must be loaded via load() before this model can be used to compute anything.
    """

    def __init__(self, settings):
        """
        Constuctor for Model

        :param settings: An object encapsulating all the settings for the fasttext-model
        :type settings: ftodtf.settings.FasttextSettings
        """
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.words_to_compare = tf.placeholder(tf.int64)
            self.embeddings = create_embedding_weights(settings)
            self.similarities = compute_word_similarities(
                self.words_to_compare, self.embeddings)
            self.saver = tf.train.Saver()

    def load(self, logdir, session):
        """ Loades pre-trained embeddings from the filesystem

        :param str logdir: The path of the folder where the checkpoints created by the training were saved
        :param tf.Session session: The session to restore the variables into
        """
        latest = tf.train.latest_checkpoint(logdir)
        print("Loading checkpoint:", latest)
        self.saver.restore(session, latest)
