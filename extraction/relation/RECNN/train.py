import tensorflow as tf
import numpy as np
import os
import datetime
import time

from text_cnn import TextCNN
import data_helpers
import utils

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def train(dataobject,**kwargs):
    with tf.device('/cpu:0'):
        sequence_length = kwargs.get('sequence_length', 100)
        max_sentence_length = kwargs.get('max_sentence_length', 90)
        dev_sample_percentage = kwargs.get('dev_sample_percentage', 0.1)
        embedding_path = kwargs.get("embedding_path", None)
        text_embedding_dim = kwargs.get("text_embedding_dim", 300)
        pos_embedding_dim = kwargs.get("pos_embedding_dim", 50)
        filter_sizes = kwargs.get("filter_sizes", "2,3,4,5")
        num_filters = kwargs.get("num_filters", 128)
        desc = kwargs.get("desc", "")
        dropout_keep_prob = kwargs.get("dropout_keep_prob", 0.5)
        l2_reg_lambda = kwargs.get("l2_reg_lambda", 1e-5)
        batch_size = kwargs.get("batch_size", 20)
        num_epochs = kwargs.get("num_epochs", 100)
        display_every = kwargs.get("display_every", 10)
        evaluate_every = kwargs.get("evaluate_every", 10)
        num_checkpoints = kwargs.get("num_checkpoints", 5)
        learning_rate = kwargs.get("learning_rate", 1.0)
        decay_rate = kwargs.get("decay_rate", 0.9)
        checkpoint_dir = kwargs.get("checkpoint_dir", "")
        allow_soft_placement = kwargs.get("allow_soft_placement", True)
        log_device_placement = kwargs.get("log_device_placement", False)
        gpu_allow_growth = kwargs.get("gpu_allow_growth", True)
        model_path = kwargs.get("model_path", "model_output")

        x_text, y, pos1, pos2 = data_helpers.load_data_and_labels(dataobject,max_sentence_length)
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sentence_length)
    x = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sentence_length)
    pos_vocab_processor.fit(pos1 + pos2)
    p1 = np.array(list(pos_vocab_processor.transform(pos1)))
    p2 = np.array(list(pos_vocab_processor.transform(pos2)))
    print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    print("position_1 = {0}".format(p1.shape))
    print("position_2 = {0}".format(p2.shape))
    print("")

    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    p1_shuffled = p1[shuffle_indices]
    p2_shuffled = p2[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = 1
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    p1_train, p1_dev = p1_shuffled[:dev_sample_index], p1_shuffled[dev_sample_index:]
    p2_train, p2_dev = p2_shuffled[:dev_sample_index], p2_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        session_conf.gpu_options.allow_growth = gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                text_vocab_size=len(text_vocab_processor.vocabulary_),
                text_embedding_size=text_embedding_dim,
                pos_vocab_size=len(pos_vocab_processor.vocabulary_),
                pos_embedding_size=pos_embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate, decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(cnn.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_path))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

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
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
            pos_vocab_processor.save(os.path.join(out_dir, "pos_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if embedding_path:
                pretrain_W = utils.load_word2vec(embedding_path, text_embedding_dim, text_vocab_processor)
                sess.run(cnn.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            print(list(zip(x_train, p1_train, p2_train, y_train)))
            batches = data_helpers.batch_iter(list(zip(x_train, p1_train, p2_train, y_train)),
                                             batch_size, num_epochs)
            # Training loop. For each batch...
            best_f1 = -1.0  # For save checkpoint(model)
            for batch in batches:
                x_batch, p1_batch, p2_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    cnn.input_text: x_batch,
                    cnn.input_p1: p1_batch,
                    cnn.input_p2: p2_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        cnn.input_text: x_dev,
                        cnn.input_p1: p1_dev,
                        cnn.input_p2: p2_dev,
                        cnn.input_y: y_dev,
                        cnn.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, 19)), average="macro")
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(f1))

                    # Model checkpoint
                    if best_f1 < f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()