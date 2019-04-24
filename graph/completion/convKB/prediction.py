import tensorflow as tf
import numpy as np
from scipy.stats import rankdata

np.random.seed(1234)
import os
from builddata import *
from model import ConvKB

def predict(test, embedding, embedding_dim, entity_array, options):
    # Parameters
    # ==================================================
    run_folder = "./"
    if embedding_dim == 50:
        model_name = "wn18"
        num_filters = 500
        useConstantInit = False
    else:
        embedding_dim = 100
        model_name = "fb15k"
        num_filters = 50
        useConstantInit = True

    head_or_tail = options.get("head_or_tail", "head")
    filter_sizes = options.get("filter_sizes", "1")
    l2_reg_lambda = options.get("l2_reg_lambda", 0.001)
    is_trainable = options.get("is_trainable", True)
    batch_size = options.get("batch_size", 128)
    neg_ratio = options.get("neg_ratio", 1.0)
    allow_soft_placement = options.get("allow_soft_placement", True)
    log_device_placement = options.get("log_device_placement", False)
    sequence_length = options.get("sequence_length")
    num_classes = options.get("num_classes")
    num_splits = options.get("num_splits", 1)
    testIdx = options.get("testIdx", 0)
    model_index = options.get("model_index", 200)
    len_words_indexes = options.get("len_words_indexes")
    len_entity2id = options.get("len_entity2id")

    x_test = np.array(list(test.keys())).astype(np.int32)
    y_test = np.array(list(test.values())).astype(np.float32)
    len_test = len(x_test)
    batch_test = len_test #int(len_test / (num_splits - 1))


    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                      log_device_placement=log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            global_step = tf.Variable(0, name="global_step", trainable=False)

            cnn = ConvKB(
                sequence_length=sequence_length,  # 3
                num_classes=num_classes,  # 1
                pre_trained=embedding,
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                vocab_size=len_words_indexes,
                l2_reg_lambda=l2_reg_lambda,
                is_trainable=is_trainable,
                useConstantInit=useConstantInit)

            # Output directory for models and summaries
            out_dir = os.path.abspath(os.path.join(run_folder, "runs", model_name))
            print("predicting {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")


            _file = checkpoint_prefix + "-" + str(model_index)

            cnn.saver.restore(sess, _file)

            print("Loaded model", _file)

            # Predict function to predict scores for test data
            def get_predict(x_batch, y_batch, writer=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                }
                cnn.predictions
                scores = sess.run([cnn.predictions], feed_dict)
                return scores


            def test_prediction(x_batch, y_batch, head_or_tail='head'):

                output = []
                for i in range(len(x_batch)):
                    if i > 0 and i%(len(x_batch)//4) == 0:
                        print(i, "/", len(x_batch), "for", head_or_tail) 
                                               
                    new_x_batch = np.tile(x_batch[i], (len_entity2id, 1))
                    new_y_batch = np.tile(y_batch[i], (len_entity2id, 1))
                    if head_or_tail == 'head':
                        new_x_batch[:, 0] = entity_array
                    else:  # 'tail'
                        new_x_batch[:, 2] = entity_array

                    lstIdx = []
                    for tmpIdxTriple in range(len(new_x_batch)):
                        tmpTriple = (new_x_batch[tmpIdxTriple][0], new_x_batch[tmpIdxTriple][1],
                                     new_x_batch[tmpIdxTriple][2])
                        if (tmpTriple in test): #(tmpTriple in train) or (tmpTriple in valid) or (tmpTriple in test): #also remove the valid test triple
                            lstIdx.append(tmpIdxTriple)
                    new_x_batch = np.delete(new_x_batch, lstIdx, axis=0)
                    new_y_batch = np.delete(new_y_batch, lstIdx, axis=0)

                    #thus, insert the valid test triple again, to the beginning of the array
                    new_x_batch = np.insert(new_x_batch, 0, x_batch[i], axis=0) #thus, the index of the valid test triple is equal to 0
                    new_y_batch = np.insert(new_y_batch, 0, y_batch[i], axis=0)

                    results = []
                    listIndexes = range(0, len(new_x_batch), (int(neg_ratio) + 1) * batch_size)
                    for tmpIndex in range(len(listIndexes) - 1):
                        results = np.append(results, get_predict(
                            new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
                            new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]]))
                    results = np.append(results,
                                        get_predict(new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:]))

                    results = np.reshape(results, -1)
                    results_with_id = rankdata(results, method='ordinal')
                    output.append(np.argmin(results_with_id))

                if head_or_tail == 'head':
                    output = np.append(np.array(output).reshape(len(output),1), np.delete(x_batch, 0, axis=1), axis=1)
                else:
                    output = np.append(np.delete(x_batch, 2, axis=1), np.array(output).reshape(len(output),1), axis=1)

                return output


            results = test_prediction(x_test, y_test, head_or_tail=head_or_tail)


    assert len(test) == len(results)

    return results
