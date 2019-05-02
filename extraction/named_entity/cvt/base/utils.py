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

"""Various utilities."""





import pickle
import sys

import tensorflow as tf


class Memoize(object):
    def __init__(self, f):
        self.f = f
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.f(*args)
        return self.cache[args]


def load_pickle(path, memoized=True):
    return _load_pickle_memoize(path) if memoized else _load_pickle(path)


def _load_pickle(path):
    with tf.gfile.GFile(path, 'rb') as f:
        return pickle.load(f)


@Memoize
def _load_pickle_memoize(path):
    return _load_pickle(path)


def write_pickle(o, path):
    tf.gfile.MakeDirs(path.rsplit('/', 1)[0])
    with tf.gfile.GFile(path, 'wb') as f:
        pickle.dump(o, f, -1)


def log(*args):
    msg = ' '.join(map(str, args))
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def heading(*args):
    log()
    log(80 * '=')
    log(*args)
    log(80 * '=')
