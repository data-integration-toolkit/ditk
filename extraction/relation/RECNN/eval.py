import tensorflow as tf
import numpy as np
import os
import subprocess

import data_helpers
import utils


def eval(dataobject,**kwargs):
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
        checkpoint_dir = kwargs.get("model_path", "runs/model_output/checkpoints/")
        allow_soft_placement = kwargs.get("allow_soft_placement", True)
        log_device_placement = kwargs.get("log_device_placement", False)
        gpu_allow_growth = kwargs.get("gpu_allow_growth", True)
        x_text, y, pos1, pos2 = data_helpers.load_data_and_labels(dataobject,max_sentence_length)

    # Map data into vocabulary
    text_path = os.path.join(checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))

    # Map data into position
    position_path = os.path.join(checkpoint_dir, "..", "pos_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
    p1 = np.array(list(position_vocab_processor.transform(pos1)))
    p2 = np.array(list(position_vocab_processor.transform(pos2)))

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        session_conf.gpu_options.allow_growth = gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
            input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(zip(x, p1, p2)), batch_size, 1, shuffle=False)

            # Collect the predictions here
            preds = []
            for batch in batches:
                x_batch, p1_batch, p2_batch = zip(*batch)
                pred = sess.run(predictions, {input_text: x_batch,
                                              input_p1: p1_batch,
                                              input_p2: p2_batch,
                                              dropout_keep_prob: 1.0})
                preds.append(pred)
            preds = np.concatenate(preds)
            truths = np.argmax(y, axis=1)

            prediction_path = os.path.join(checkpoint_dir, "..", "predictions.txt")
            truth_path = os.path.join(checkpoint_dir, "..", "ground_truths.txt")
            prediction_file = open(prediction_path, 'w')
            truth_file = open(truth_path, 'w')
            for i in range(len(preds)):
                prediction_file.write("{}\t{}\n".format(i, utils.label2class[preds[i]]))
                truth_file.write("{}\t{}\n".format(i, utils.label2class[truths[i]]))
            prediction_file.close()
            truth_file.close()

            perl_path = os.path.join(os.path.curdir,
                                     "SemEval2010_task8_all_data",
                                     "SemEval2010_task8_scorer-v1.2",
                                     "semeval2010_task8_scorer-v1.2.pl")
            process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
            for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
                print(line)
                
def load_model(**kwargs):
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
        checkpoint_dir = kwargs.get("model_path", "runs/model_output/checkpoints/")
        allow_soft_placement = kwargs.get("allow_soft_placement", True)
        log_device_placement = kwargs.get("log_device_placement", False)
        gpu_allow_growth = kwargs.get("gpu_allow_growth", True)
  
    # Map data into vocabulary
    text_path = os.path.join(checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    # Map data into position
    position_path = os.path.join(checkpoint_dir, "..", "pos_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
   
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        session_conf.gpu_options.allow_growth = gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
            input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    return graph,sess

# def main(_):
#     eval()


# if __name__ == "__main__":
#     tf.app.run()