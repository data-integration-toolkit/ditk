""" This module handles the expporting of trained models"""
import os.path
import tensorflow as tf
import ftodtf.model as model


def export_embeddings(settings, outputdir):
    """ Builds an model using the given settings, loads the last checkpoint and saves only the embedding-variable to a new checkpoint inside outputdir,
        leaving out all the other weights. The new checkpoint is much smaller then the original.
        This new Checkpoint can be used for inference but not to continue training.

        :param ftodtf.settings.FasttextSettings settings: The settings for the model
        :param str outputdir: The directory to store the new checkpoint to.
    """
    m = model.InferenceModel(settings)
    sess = tf.Session(graph=m.graph)
    m.load(settings.log_dir, sess)
    with m.graph.as_default():
        exporter = tf.train.Saver(
            save_relative_paths=True, var_list=m.embeddings, filename="embeddings")
        exporter.save(sess, os.path.join(outputdir, "embeddings"))
