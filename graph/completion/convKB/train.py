import tensorflow as tf
import numpy as np
import os
np.random.seed(1234)
from graph.completion.convKB.builddata import *
from graph.completion.convKB.model import ConvKB

def train(train_batch, embedding, embedding_dim, options):

    # Parameters
    # ==================================================
    ditk_path = ""
    for path in sys.path:
        if "ditk" in path and not "graph" in path:
            ditk_path = path
    run_folder = ditk_path + "/graph/completion/convKB/"

    if embedding_dim == 50:
        model_name = "wn18"
        num_filters = 500
        learning_rate = 0.0001
        useConstantInit = False
    else:
        model_name = "fb15k"
        num_filters = 50
        learning_rate = 0.000005
        useConstantInit = True

    filter_sizes = options.get("filter_sizes", "1")
    dropout_keep_prob = options.get("dropout_keep_prob", 1.0)
    l2_reg_lambda = options.get("l2_reg_lambda", 0.001)
    is_trainable = options.get("is_trainable", True)
    batch_size = options.get("batch_size", 128)
    neg_ratio = options.get("neg_ratio", 1.0)
    num_epochs = options.get("num_epochs", 21)
    save_step = options.get("save_step", 20)
    allow_soft_placement = options.get("allow_soft_placement", True)
    log_device_placement = options.get("log_device_placement", False)
    data_size = options.get("data_size")
    sequence_length = options.get("sequence_length")
    num_classes = options.get("num_classes")
    len_words_indexes = options.get("len_words_indexes")

    # Training
    # ==================================================
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
                # pre_trained=[],
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                vocab_size=len_words_indexes,
                l2_reg_lambda=l2_reg_lambda,
                is_trainable=is_trainable,
                useConstantInit=useConstantInit)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            out_dir = os.path.abspath(os.path.join(run_folder, "runs", model_name))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob,
                }
                _, step, loss = sess.run([train_op, global_step, cnn.loss], feed_dict)

            num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                if epoch > 0 and epoch%(num_epochs//4) == 0:
                    print(epoch,"/",num_epochs)
                
                for batch_num in range(num_batches_per_epoch):
                    x_batch, y_batch = train_batch()
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)

                if epoch > 0:
                    if epoch % save_step == 0:
                        path = cnn.saver.save(sess, checkpoint_prefix, global_step=epoch)
                        print("Saved model checkpoint to {}\n".format(path))

    return cnn