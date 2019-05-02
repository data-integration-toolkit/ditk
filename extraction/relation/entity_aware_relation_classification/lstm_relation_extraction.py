# Python 3.x

import os
import numpy as np
import pandas as pd
import tensorflow as tf

import data_helpers
from configure import FLAGS
from logger import Logger
from model import EntityAttentionLSTM
import utils
import datetime

from relation_extraction import RelationExtraction

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class LSTM_relation_extraction(RelationExtraction):
    train_data = {}
    test_data = {}
    sess = None
    vocab_processor = None
    pos_vocab_processor = None

    def __init__(self):
        RelationExtraction.__init__(self)

        self.train_data = {}
        self.test_data = {}
        self.sess = None
        self.vocab_processor = None
        self.pos_vocab_processor = None

    def read_dataset(self, input_file, *args, **kwargs):
        """
        Reads a dataset to be used for training

         Note: The child file of each member overrides this function to read dataset
         according to their data format.

        Args:
            input_file: Filepath with list of files to be read
        Returns:
            (optional):Data from file
        """
        # for key, value in input_file.items():
        #     if key == 'train':
        train_file_path = input_file["train"]
        test_file_path = input_file["test"]
        train_text, train_y, train_e1, train_e2, train_pos1, train_pos2, train_relation = \
            data_helpers.load_data_from_common_data(train_file_path, 1, 0, FLAGS.data_type)
        self.train_data = {
            "text": train_text,
            "y": train_y,
            "e1": train_e1,
            "e2": train_e2,
            "pos1": train_pos1,
            "pos2": train_pos2,
            "relation": train_relation
        }

        test_text, test_y, test_e1, test_e2, test_pos1, test_pos2, test_relation = \
            data_helpers.load_data_from_common_data(test_file_path, 8001, train_y.shape[1], FLAGS.data_type)
        self.test_data = {
            "text": test_text,
            "y": test_y,
            "e1": test_e1,
            "e2": test_e2,
            "pos1": test_pos1,
            "pos2": test_pos2,
            "relation": test_relation
        }

        # Build vocabulary
        # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
        # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
        # =>
        # [27 39 40 41 42  1 43  0  0 ... 0]
        # dimension = MAX_SENTENCE_LENGTH
        self.vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
        self.vocab_processor.fit(train_text + test_text)
        self.train_data["x"] = np.array(list(self.vocab_processor.transform(train_text)))
        self.test_data["x"] = np.array(list(self.vocab_processor.transform(test_text)))
        self.train_data["text"] = np.array(train_text)
        self.test_data["text"] = np.array(test_text)
        print("\nText Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
        print("train_x = {0}".format(self.train_data["x"].shape))
        print("train_y = {0}".format(self.train_data["y"].shape))
        print("test_x = {0}".format(self.test_data["x"].shape))
        print("test_y = {0}".format(self.test_data["y"].shape))

        # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
        # [95 96 97 98 99 100 101 999 999 999 ... 999]
        # =>
        # [11 12 13 14 15  16  21  17  17  17 ...  17]
        # dimension = MAX_SENTENCE_LENGTH
        self.pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
        self.pos_vocab_processor.fit(train_pos1 + train_pos2 + test_pos1 + test_pos2)
        self.train_data["p1"] = np.array(list(self.pos_vocab_processor.transform(train_pos1)))
        self.train_data["p2"] = np.array(list(self.pos_vocab_processor.transform(train_pos2)))
        self.test_data["p1"] = np.array(list(self.pos_vocab_processor.transform(test_pos1)))
        self.test_data["p2"] = np.array(list(self.pos_vocab_processor.transform(test_pos2)))
        print("\nPosition Vocabulary Size: {:d}".format(len(self.pos_vocab_processor.vocabulary_)))
        print("train_p1 = {0}".format(self.train_data["p1"].shape))
        print("test_p1 = {0}".format(self.test_data["p1"].shape))
        print("")

        return self.train_data, self.test_data

    def data_preprocess(self, input_data, *args, **kwargs):
        """
         (Optional): For members who do not need preprocessing. example: .pkl files
         A common function for a set of data cleaning techniques such as lemmatization, count vectorizer and so forth.
        Args:
            input_data: Raw data to tokenize
        Returns:
            Formatted data for further use.
        """
        pass

    def tokenize(self, input_data, ngram_size=None, *args, **kwargs):
        """
        Tokenizes dataset using Stanford Core NLP(Server/API)
        Args:
            input_data: str or [str] : data to tokenize
            ngram_size: mention the size of the token combinations, default to None
        Returns:
            tokenized version of data
        """
        pass

    def train(self, train_data, *args, **kwargs):
        """
        Trains a model on the given training data

         Note: The child file of each member overrides this function to train data
         according to their algorithm.

        Args:
            train_data: post-processed data to be trained.

        Returns:
            (Optional) : trained model in applicable formats.
             None: if the model is stored internally.
        """

        train_text = train_data["text"]
        train_x = train_data["x"]
        train_y = train_data["y"]
        train_e1 = train_data["e1"]
        train_e2 = train_data["e2"]
        train_p1 = train_data["p1"]
        train_p2 = train_data["p2"]

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            sess = tf.Session(config=session_conf)
            self.sess = sess
            with sess.as_default():
                model = EntityAttentionLSTM(
                    sequence_length=train_x.shape[1],
                    num_classes=train_y.shape[1],
                    vocab_size=len(self.vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_size,
                    pos_vocab_size=len(self.pos_vocab_processor.vocabulary_),
                    pos_embedding_size=FLAGS.pos_embedding_size,
                    hidden_size=FLAGS.hidden_size,
                    num_heads=FLAGS.num_heads,
                    attention_size=FLAGS.attention_size,
                    use_elmo=(FLAGS.embeddings == 'elmo'),
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
                gvs = optimizer.compute_gradients(model.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                # Output directory for models and summaries
                out_dir = FLAGS.data_type+("_{}".format(FLAGS.embeddings) if 0 < len(FLAGS.embeddings) else "")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out_dir))
                print("\nWriting to {}\n".format(out_dir))

                # Logger
                logger = Logger(out_dir)

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", model.loss)
                acc_summary = tf.summary.scalar("accuracy", model.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)


                # Write vocabulary
                self.vocab_processor.save(os.path.join(out_dir, "vocab"))
                self.pos_vocab_processor.save(os.path.join(out_dir, "pos_vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                if FLAGS.embeddings == "word2vec":
                    pre_train_w = utils.load_word2vec('resource/GoogleNews-vectors-negative300.bin',
                                                      FLAGS.embedding_size,
                                                      self.vocab_processor)
                    sess.run(model.W_text.assign(pre_train_w))
                    print("Success to load pre-trained word2vec model!\n")
                elif FLAGS.embeddings == "glove100":
                    pre_train_w = utils.load_glove('resource/glove.6B.100d.txt',
                                                   FLAGS.embedding_size,
                                                   self.vocab_processor)
                    sess.run(model.W_text.assign(pre_train_w))
                    print("Success to load pre-trained glove100 model!\n")
                elif FLAGS.embeddings == "glove300":
                    pre_train_w = utils.load_glove('resource/glove.840B.300d.txt',
                                                   FLAGS.embedding_size,
                                                   self.vocab_processor)
                    sess.run(model.W_text.assign(pre_train_w))
                    print("Success to load pre-trained glove300 model!\n")

                # Generate batches
                train_batches = data_helpers.batch_iter(list(zip(train_x, train_y, train_text,
                                                                 train_e1, train_e2, train_p1, train_p2)),
                                                        FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                best_f1 = 0.0  # For save checkpoint(model)
                for train_batch in train_batches:
                    train_bx, train_by, train_btxt, train_be1, train_be2, train_bp1, train_bp2 = zip(*train_batch)
                    feed_dict = {
                        model.input_x: train_bx,
                        model.input_y: train_by,
                        model.input_text: train_btxt,
                        model.input_e1: train_be1,
                        model.input_e2: train_be2,
                        model.input_p1: train_bp1,
                        model.input_p2: train_bp2,
                        model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                        model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                    train_summary_writer.add_summary(summaries, step)

                    # Training log display
                    if step % FLAGS.display_every == 0:
                        logger.logging_train(step, loss, accuracy)

                self.save_model(checkpoint_prefix)

        return

    def predict(self, test_data, write_file=True, entity_1=None, entity_2=None, trained_model=None, *args, **kwargs):
        """

        :param test_data:
        :param write_file:
        :param entity_1:
        :param entity_2:
        :param trained_model:
        :param args:
        :param kwargs:
        :return:
        """

        test_text = test_data["text"]
        test_x = test_data["x"]
        test_y = test_data["y"]
        test_e1 = test_data["e1"]
        test_e2 = test_data["e2"]
        test_p1 = test_data["p1"]
        test_p2 = test_data["p2"]

        model_path = "runs/" + FLAGS.data_type + ("_{}".format(FLAGS.embeddings) if 0 < len(FLAGS.embeddings) else "")

        tf.Graph().as_default()
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        self.sess = sess
        sess.as_default()
        model = EntityAttentionLSTM(
            sequence_length=test_x.shape[1],
            num_classes=test_y.shape[1],
            vocab_size=len(self.vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_size,
            pos_vocab_size=len(self.pos_vocab_processor.vocabulary_),
            pos_embedding_size=FLAGS.pos_embedding_size,
            hidden_size=FLAGS.hidden_size,
            num_heads=FLAGS.num_heads,
            attention_size=FLAGS.attention_size,
            use_elmo=(FLAGS.embeddings == 'elmo'),
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Logger
        logger = Logger(model_path)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.load_model(os.path.abspath(os.path.join(model_path, "checkpoints")))
        # checkpoint_dir = os.path.abspath(os.path.join(model_path, "checkpoints"))
        #
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        test_batches = data_helpers.batch_iter(list(zip(test_x, test_y, test_text,
                                                        test_e1, test_e2, test_p1, test_p2)),
                                               FLAGS.batch_size, 1, shuffle=False)
        # Training loop. For each batch...
        losses = 0.0
        accuracy = 0.0
        predictions = []
        iter_cnt = 0
        for test_batch in test_batches:
            test_bx, test_by, test_btxt, test_be1, test_be2, test_bp1, test_bp2 = zip(*test_batch)
            feed_dict = {
                model.input_x: test_bx,
                model.input_y: test_by,
                model.input_text: test_btxt,
                model.input_e1: test_be1,
                model.input_e2: test_be2,
                model.input_p1: test_bp1,
                model.input_p2: test_bp2,
                model.emb_dropout_keep_prob: 1.0,
                model.rnn_dropout_keep_prob: 1.0,
                model.dropout_keep_prob: 1.0
            }
            loss, acc, pred = sess.run(
                [model.loss, model.accuracy, model.predictions], feed_dict)
            losses += loss
            accuracy += acc
            predictions += pred.tolist()
            iter_cnt += 1
        losses /= iter_cnt
        accuracy /= iter_cnt
        predictions = np.array(predictions, dtype='int')

        # logger.logging_eval(1, loss, accuracy, predictions)

        res = []
        timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        prediction_path = os.path.abspath(os.path.join(logger.log_dir, "predictions_{0}.txt".format(timestamp)))
        if write_file:
            f = open(prediction_path, "w")

        for i in range(len(predictions)):
            sentence = self.test_data["text"][i]
            tokens = sentence.split(" ")

            sentence = sentence.replace("e11 ", "").replace("e12 ", "").replace("e21 ", "").replace(" e22", "")
            e1 = tokens[self.test_data["e1"][i]]
            e2 = tokens[self.test_data["e2"][i]]
            truth = utils.label2class[FLAGS.data_type][self.test_data["relation"][i]]
            prediction = utils.label2class[FLAGS.data_type][predictions[i]]

            res.append([sentence, e1, e2, predictions[i], self.test_data["relation"][i]])
            # res.append([sentence, e1, e2, prediction, truth])
            if write_file:
                f.writelines('\t'.join([sentence, e1, e2, prediction, truth]) + '\n')

        if write_file:
            f.close()

        return res, prediction_path

    def evaluate(self, input_data, prediction_data=None, trained_model=None, *args, **kwargs):
        """
        Evaluates the result based on the benchmark dataset and the evauation metrics  [Precision,Recall,F1, or others...]
         Args:
             input_data: benchmark dataset/evaluation data
             trained_model: trained model or None if stored internally
        Returns:
            performance metrics: tuple with (p,r,f1) or similar...
        """
        if prediction_data == None:
            data, _ = self.predict(input_data, False)
        else:
            data = prediction_data
        df = pd.DataFrame(data=data, columns=["sentence", "e1", "e2", "prediction", "truth"])

        tp = 0
        fp = 0
        fn = 0

        prediction = df['prediction'].tolist()
        truth = df['truth'].tolist()
        for i in range(len(prediction)):
            if truth[i] == 0 and prediction[i] != 0:
                fn = fn + 1
            elif truth[i] != 0:
                if truth[i] == prediction[i]:
                    tp = tp + 1
                else:
                    fp = fp + 1

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2 * ((precision*recall)/(precision+recall))

        f1_log = "Precision: {:.2f}\nRecall: {:.2f}\nF1-score: {:.2f}%".format(precision, recall, f1*100)
        print(f1_log)

        return [precision, recall, f1]

    def save_model(self, model_path, *args, **kwargs):
        """
            :param model_path: From where to save the model - Optional function
            :return:
        """
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        saver.save(self.sess, model_path)

    def load_model(self, model_path, *args, **kwargs):
        """
            :param model_path: From where to load the model - Optional function
            :return:
        """
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path))


if __name__ == "__main__":
    FLAGS.embeddings = 'glove300'
    FLAGS.data_type = 'nyt'

    files_dict = {
        "train": "data/" + FLAGS.data_type + "/trainfile.txt",
        "test": "data/" + FLAGS.data_type + "/testfile.txt"
    }

    my_model = LSTM_relation_extraction()
    train_data, test_data = my_model.read_dataset(files_dict)
    my_model.train(train_data)
    predictions, output_file_path = my_model.predict(test_data)
    my_model.evaluate(None, predictions)

