import utils
import tf_utils
from build_data import build_data
import numpy as np
import tensorflow as tf
import sys
import os.path


def checkInputs():
    if (len(sys.argv) <= 3) or os.path.isfile(sys.argv[0]) == False:
        raise ValueError(
            'The configuration file and the timestamp should be specified.')

    es_file = sys.argv[3] + "/es_" + sys.argv[2] + ".txt"
    es_epoch = sys.maxsize
    if os.path.isfile(es_file) == True:
        with open(es_file, 'r') as myfile:
            es_epoch = int(myfile.read())
            myfile.close()
    return es_epoch


if __name__ == "__main__":

    es_epoch = checkInputs()
    config = build_data(sys.argv[1])
    config.train_id_docs.extend(config.dev_id_docs)
    train_data = utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))
    test_data = utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))
    tf.reset_default_graph()
    tf.set_random_seed(1)
    utils.printParameters(config)

    # ---- Training ----
    config1 = tf.ConfigProto()
    config1.gpu_options.per_process_gpu_memory_fraction = 0.85
    with tf.Session(config=config1) as sess:
        # saver = tf.train.import_meta_graph('model.ckpt.meta')
        # saver.restore(sess, 'model.ckpt')
        embedding_matrix = tf.get_variable('embedding_matrix', shape=config.wordvectors.shape, dtype=tf.float32,
                                           trainable=False).assign(config.wordvectors)
        emb_mtx = sess.run(embedding_matrix)
        model = tf_utils.model(config, emb_mtx, sess)
        obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel = model.run()

        train_step = model.get_train_op(obj)
        operations = tf_utils.operations(train_step, obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel,
                                         actual_op_rel, score_op_rel)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping

        for iter in range(config.nepochs + 1):
            model.train(train_data, operations, iter)
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in path: %s" % save_path)

            # ---- Testing ----
        print("Starting Testing\n")

        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(sess, 'model.ckpt')
        for iter in range(config.nepochs + 1):
            test_score = model.evaluate(test_data, operations, 'test')
            print("\n- Test score {} in {} epoch\n".format(test_score, iter))