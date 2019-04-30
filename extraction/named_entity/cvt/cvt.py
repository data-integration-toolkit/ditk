# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Run training and evaluation for CVT text models."""


import tensorflow as tf

from os import getcwd
from extraction.named_entity.cvt.base import configure
from extraction.named_entity.cvt.base import utils
from extraction.named_entity.cvt.training import trainer
from extraction.named_entity.cvt.training import training_progress


cwd = getcwd()


def main(mode='train', model_name='chunking_model', data_dir='/mini_data', size='mini', gdrive_mounted='f'):
    utils.heading('SETUP')
    if size == 'mini':
        args_dict = {'warm_up_steps': 50.0, 'train_batch_size': 10, 'test_batch_size': 10, 'data_dir': data_dir,
                     'mode': mode, 'model_name': model_name, 'eval_dev_every': 75, 'eval_train_every': 150,
                     'save_model_every': 10}
        config = configure.Config(**args_dict)
    else:
        config = configure.Config(data_dir=data_dir, mode=mode, model_name=model_name)
    config.write()
    with tf.Graph().as_default() as graph:
        model_trainer = trainer.Trainer(config)
        summary_writer = tf.summary.FileWriter(config.summaries_dir)
        checkpoints_saver = tf.train.Saver(max_to_keep=1)
        best_model_saver = tf.train.Saver(max_to_keep=1)
        init_op = tf.global_variables_initializer()
        graph.finalize()
        with tf.Session() as sess:
            sess.run(init_op)
            progress = training_progress.TrainingProgress(
                config, sess, checkpoints_saver, best_model_saver,
                config.mode == 'train')
            utils.log()
            if config.mode == 'train':
                utils.heading('START TRAINING ({:})'.format(config.model_name))
                model_trainer.train(sess, progress, summary_writer)
            elif config.mode == 'eval':
                utils.heading('RUN EVALUATION ({:})'.format(config.model_name))
                progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
                    config.checkpoints_dir))
                op_preds = model_trainer.evaluate_all_tasks(sess, summary_writer, None)
                return op_preds
            else:
                raise ValueError('Mode must be "train" or "eval"')


def main_funct(mode='train', model_name='chunking_model', data_dir='/mini_data', size='mini', gdrive_mounted='f'):
    data_folder = data_dir
    if size == 'mini':
        op_preds = main(data_dir=data_folder, mode=mode, model_name=model_name, size=size,
                        gdrive_mounted=gdrive_mounted)
    else:
        op_preds = main(data_dir=data_folder, mode=mode, model_name=model_name, size=size,
                        gdrive_mounted=gdrive_mounted)
    return op_preds


if __name__ == '__main__':
    main_funct()


