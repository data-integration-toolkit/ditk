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

"""
Preprocesses pretrained word embeddings, creates dev sets for tasks without a
provided one, and figures out the set of output classes for each task.
"""


import os
import random
from extraction.named_entity.cvt.base import configure
from extraction.named_entity.cvt.base import embeddings
from extraction.named_entity.cvt.base import utils
from extraction.named_entity.cvt.task_specific.word_level import word_level_data

cwd = os.getcwd()  # elsewhere os is being used. So have to import entire os and not just getcwd


def main(data_dir='./data'):
    random.seed(0)
    utils.log("BUILDING WORD VOCABULARY/EMBEDDINGS")
    for pretrained in ['glove.6B.300d.txt']:
        config = configure.Config(data_dir=data_dir,
                                  for_preprocessing=True,
                                  pretrained_embeddings=pretrained,
                                  word_embedding_size=300)
        embeddings.PretrainedEmbeddingLoader(config).build()

    utils.log("CONSTRUCTING DEV SETS")
    for task_name in ["chunk"]:
        # chunking does not come with a provided dev split, so create one by
        # selecting a random subset of the data
        config = configure.Config(data_dir=data_dir,
                                  for_preprocessing=True)
        task_data_dir = os.path.join(config.raw_data_topdir, task_name) + '/'
        train_sentences = word_level_data.TaggedDataLoader(
            config, task_name, False).get_labeled_sentences("train")
        random.shuffle(train_sentences)
        if 'mini_data' not in data_dir:
            write_sentences(task_data_dir + 'train_subset.txt', train_sentences[1500:])
            write_sentences(task_data_dir + 'dev.txt', train_sentences[:1500])
        else:
            write_sentences(task_data_dir + 'train_subset.txt', train_sentences[len(train_sentences)//4:])
            write_sentences(task_data_dir + 'dev.txt', train_sentences[:len(train_sentences)//4])

    utils.log("WRITING LABEL MAPPINGS")
    for task_name in ["chunk"]:
        for i, label_encoding in enumerate(["BIOES"]):
            config = configure.Config(data_dir=data_dir,
                                      for_preprocessing=True,
                                      label_encoding=label_encoding)
            token_level = task_name in ["ccg", "pos", "depparse"]
            loader = word_level_data.TaggedDataLoader(config, task_name, token_level)
            if token_level:
                if i != 0:
                    continue
                utils.log("WRITING LABEL MAPPING FOR", task_name.upper())
            else:
                utils.log("  Writing label mapping for", task_name.upper(),
                          label_encoding)
            utils.log(" ", len(loader.label_mapping), "classes")
            utils.write_pickle(loader.label_mapping,
                               loader.label_mapping_path)


def write_sentences(fname, sentences):
    with open(fname, 'w') as f:
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                f.write(word + " " + tag + "\n")
            f.write("\n")


def main_funct_pre(data_dir='/mini_data', size='mini', gdrive_mounted='f'):
    data_folder = data_dir
    main(data_dir=data_folder)


if __name__ == '__main__':
    main_funct_pre()
