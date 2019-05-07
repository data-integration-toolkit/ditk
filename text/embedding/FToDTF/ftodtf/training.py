#!/usr/env python3

# Copyright 2018 The FToDTF Authors.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ATTENTION: the original file (found at https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
# has been completely modified by the FToDTF Authors!
#
#
# ==============================================================================
""" This module handles the training of the word-vectors"""
import os
import signal
import _thread

# pylint: disable=E0611
from tensorflow.python.client import timeline
import tensorflow as tf

import ftodtf.model as model
import ftodtf.inference


def _tfserver_from_settings(settings):
    """ Given a settings-object creates and returns a tf-server with proper settings

    :param settings: An object encapsulating all the settings for the fasttext-model
    :type settings: ftodtf.settings.FasttextSettings
    :returns: A tf.train.Server configured and ready to run and the cluster-spec that was used to create the server
    """
    spec = tf.train.ClusterSpec({
        "worker": settings.workers_list,
        "ps": settings.ps_list
    })

    server = tf.train.Server(
        spec, job_name=settings.job, task_index=settings.index)
    return server, spec


class PrintLossHook(tf.train.StepCounterHook):
    """ Implements a Hook that prints the current step and current average loss every x steps"""

    def __init__(self, every_n_steps, lossop, steptensor):
        self.lossop = lossop
        self.steptensor = steptensor
        self.cumloss = 0
        self.every_n_steps = every_n_steps
        self.stepcounter = 0
        super().__init__(self)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.lossop, self.steptensor])

    def after_run(self, run_context, run_values):
        loss, step = run_values.results
        if self.stepcounter == self.every_n_steps:
            print("Step {}: Loss: {}".format(
                step, self.cumloss/(self.stepcounter)))
            self.cumloss = 0
            self.stepcounter = 1
        else:
            self.cumloss += loss
            self.stepcounter += 1


def train(settings):
    """ Run the fasttext training.

    :param settings: An object encapsulating all the settings for the fasttext-model
    :type settings: ftodtf.settings.FasttextSettings

    """
    if not os.path.exists(settings.log_dir):
        os.makedirs(settings.log_dir)

    server, cluster = _tfserver_from_settings(settings)

    if settings.job == "ps":
        _thread.start_new_thread(server.join, tuple())
        signal.sigwait([signal.SIGINT, signal.SIGKILL])
        print("Terminating...")
        return

    # Get the computation-graph and the associated operations
    m = model.TrainingModel(settings, cluster)

    hooks = [tf.train.StopAtStepHook(
        last_step=settings.steps)]

    chief_hooks = [PrintLossHook(2000, m.loss, m.step_nr)]

    if settings.validation_words:
        chief_hooks.append(ftodtf.inference.PrintSimilarityHook(
            10000, m.validation, settings.validation_words_list))

    with m.graph.as_default():
        with tf.train.MonitoredTrainingSession(
                scaffold=m.get_scaffold(),
                master=server.target,
                is_chief=(settings.index == 0),
                checkpoint_dir=settings.log_dir,
                hooks=hooks,
                save_checkpoint_steps=10000,
                chief_only_hooks=chief_hooks) as session:

            # Open a writer to write summaries.
            while not session.should_stop():

                # Define metadata variable.
                run_metadata = tf.RunMetadata()
                options = None
                if settings.profile:
                    # pylint: disable=E1101
                    options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                session.run(
                    [m.optimizer, m.merged, m.loss],
                    run_metadata=run_metadata,
                    options=options)

                # Create the Timeline object, and write it to a json file
                if settings.profile:
                    # pylint: disable=E1101
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('profile.json', 'w') as f:
                        f.write(chrome_trace)
