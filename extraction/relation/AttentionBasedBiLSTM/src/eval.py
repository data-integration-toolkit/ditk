from time import time

import tensorflow as tf
import numpy as np
import os

from src import data_helpers, utils


def restoreModel(filePath,**kwargs):

    if(filePath==''):
        checkpoint_file = tf.train.latest_checkpoint(kwargs.get('checkpoint_dir',"./runs/models/checkpoints"))
    else:
        checkpoint_file = tf.train.latest_checkpoint(filePath)

    text_path = os.path.join(kwargs.get('checkpoint_dir'), "..", "vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=kwargs.get('allow_soft_placement', True),
            log_device_placement=kwargs.get('log_device_placement', False))
        session_conf.gpu_options.allow_growth = kwargs.get('gpu_allow_growth', True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)


    return sess,text_vocab_processor,graph



def predictFromModel(data,sess_restored,text_vocab_processor,graph,**kwargs):


    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(data, 'test')

    # Map data into vocabulary
    # text_path = os.path.join(kwargs.get('checkpoint_dir'), "..", "vocab")
    # text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))

    checkpoint_file = tf.train.latest_checkpoint(kwargs.get('checkpoint_dir',"../runs/models/checkpoints"))



    with graph.as_default():
        # session_conf = tf.ConfigProto(
        #     allow_soft_placement=kwargs.get('allow_soft_placement',True),
        #     log_device_placement=kwargs.get('log_device_placement',False))
        # session_conf.gpu_options.allow_growth = kwargs.get('gpu_allow_growth',True)
        # sess = tf.Session(config=session_conf)
        # with sess.as_default():
        #     # Load the saved meta graph and restore variables
        #     saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        #     saver.restore(sess, checkpoint_file)
            # sess = sess_restored
            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x), kwargs.get('batch_size', 10), 1, shuffle=False)

            # Collect the predictions here
            preds = []
            for x_batch in batches:
                pred = sess_restored.run(predictions, {input_text: x_batch,
                                              emb_dropout_keep_prob: 1.0,
                                              rnn_dropout_keep_prob: 1.0,
                                              dropout_keep_prob: 1.0})
                preds.append(pred)
            preds = np.concatenate(preds)
            truths = np.argmax(y, axis=1)



            prediction_path = os.path.join(kwargs.get('checkpoint_dir'), "..", "predictions.txt")
            truth_path = os.path.join(kwargs.get('checkpoint_dir'), "..", "ground_truths.txt")
            prediction_file = open(prediction_path, 'w')
            truth_file = open(truth_path, 'w')
            for i in range(len(preds)):
                prediction_file.write("{}\t{}\n".format(i, utils.label2class[preds[i]]))
                truth_file.write("{}\t{}\n".format(i, utils.label2class[truths[i]]))
            prediction_file.close()
            truth_file.close()


    return preds,truths,prediction_path,truth_path


def eval(data,**kwargs):


    tensforFlow_session,text_vocab_processor,graph= restoreModel('',**kwargs)

    preds,truths,prediction_path,truth_path = predictFromModel(data,tensforFlow_session,text_vocab_processor,graph,**kwargs)

    micro, macro, weighted = tf_f1_score(truths, preds)
    print("\n")

    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        stime = time()
        mic, mac, wei = sess.run([micro, macro, weighted])


    # perl_path = os.path.join(os.path.curdir,"data",
    #                          "SemEval2010_task8_all_data",
    #                          "SemEval2010_task8_scorer-v1.2",
    #                          "semeval2010_task8_scorer-v1.2.pl")
    # process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
    # for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
    #     print(line)




def predict(data,**kwargs):

    print("Loading model for predicting............")

    tensforFlow_session,text_vocab_processor,graph= restoreModel('', **kwargs)

    preds, truths, prediction_path, truth_path = predictFromModel(data,tensforFlow_session,text_vocab_processor,graph,**kwargs)

    output_data =[]

    # sentence e1 e2 predictedData TruthData

    print("\n\n")


    with open("../res/output_predictions.txt", "w") as f:
        for lineNumber in range(0,len(data)):
            line = data[lineNumber]
            line[1] = line[1].replace('<e1>','')
            line[1] = line[1].replace('</e1>','')
            line[1] = line[1].replace('<e2>', '')
            line[1] = line[1].replace('</e2>', '')

            output_row = [line[1], line[2], line[5], utils.label2class[int(preds[lineNumber])], line[8]]


            output_data.append(output_row)

            f.writelines("\t".join([line[1], line[2], line[5], utils.label2class[int(preds[lineNumber])], line[8]]) + '\n')

            # printing sentence and predictions

            print("Sentence :- ",line[1])
            print("Entity 1 :- ",line[2])
            print("Entity 2 :- ",line[5])
            print("Predicted Relation :- ", utils.label2class[int(preds[lineNumber])])


    #get model and then predict sentences

    output_file_path = '../res/output_predictions.txt'


    return output_data,output_file_path

def tf_f1_score(y_true, y_pred):
    """Computes 3 different f1 scores, micro macro
    weighted.
    micro: f1 score accross the classes, as 1
    macro: mean of f1 scores per class
    weighted: weighted average of f1 scores per class,
            weighted from the support of each class


    Args:
        y_true (Tensor): labels, with shape (batch, num_classes)
        y_pred (Tensor): model's predictions, same shape as y_true

    Returns:
        tuple(Tensor): (micro, macro, weighted)
                    tuple of the computed f1 scores
    """

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        with tf.Session() as sess:
            tf.global_variables_initializer().run(session=sess)
            stime = time()
            mic, mac, wei = sess.run([precision, recall, f1])
        print(
            ' Precision: {:.8f}\t    Recall: {:.8f}\t    F1: {:.8f}'.format(
                mic, mac, wei
            ))




        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted




