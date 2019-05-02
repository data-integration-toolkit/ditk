import tensorflow as tf
import numpy as np
import os
import datetime
import time

from src.att_lstm import AttLSTM
from src import data_helpers, utils

from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")


def train(data,**kwargs):

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(data, 'train')

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(kwargs.get('max_sentence_length',90))
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))


    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(kwargs.get('dev_sample_percentage',0.1) * float(len(y)))
    if(dev_sample_index ==0):
        dev_sample_index = -1
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("\n\nTrain/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=kwargs.get('allow_soft_placement',True),
            log_device_placement=kwargs.get('log_device_placement',False))
        session_conf.gpu_options.allow_growth = kwargs.get('gpu_allow_growth',True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AttLSTM(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=kwargs.get('embedding_dim',100),
                hidden_size=kwargs.get('hidden_size',100),
                l2_reg_lambda=kwargs.get('l2_reg_lambda',1e-5))

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(kwargs.get('learning_rate',1.0), kwargs.get('decay_rate',0.9), 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "../runs", "models"))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=kwargs.get('num_checkpoints',5))

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if 'embedding_path' in kwargs:
                pretrain_W = utils.load_glove(kwargs.get('embedding_path', '../res/glove.6B.100d.txt'), kwargs.get('embedding_dim', 100), vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), kwargs.get('batch_size', 10), kwargs.get('num_epochs', 100))
            # Training loop. For each batch...
            best_f1 = -1.0  # For save checkpoint(model)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    model.input_text: x_batch,
                    model.input_y: y_batch,
                    model.emb_dropout_keep_prob: kwargs.get('emb_dropout_keep_prob',0.7),
                    model.rnn_dropout_keep_prob: kwargs.get('rnn_dropout_keep_prob',0.7),
                    model.dropout_keep_prob: kwargs.get('dropout_keep_prob',0.5)
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % kwargs.get('display_every',10) == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % kwargs.get('evaluate_every',10) == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        model.input_text: x_dev,
                        model.input_y: y_dev,
                        model.emb_dropout_keep_prob: 1.0,
                        model.rnn_dropout_keep_prob: 1.0,
                        model.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, len(
                        utils.class2label))), average="macro")
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    # print(" F1 Score : {:g}\n".format(f1))

                    # Model checkpoint
                    if best_f1 < f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))
