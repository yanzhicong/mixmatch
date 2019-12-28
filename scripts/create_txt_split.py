#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to create SSL splits from a dataset.
"""

from collections import defaultdict
import json
import os
import sys
sys.path.append('./')

# os.path.append('../')


from absl import app
from absl import flags
from libml import utils
from libml.data import DATA_DIR
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from typing import List, Any

flags.DEFINE_integer('seed', 0, 'Random seed to use, 0 for no shuffling.')
flags.DEFINE_integer('size', 0, 'Size of labelled set.')

FLAGS = flags.FLAGS

def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']


def main(argv):
    assert FLAGS.size
    argv.pop(0)

    if any(not os.path.exists(f) for f in argv[1:]):
        raise FileNotFoundError(argv[1:])

    target = '%s.%d@%d' % (argv[0], FLAGS.seed, FLAGS.size)
    if os.path.exists(target):
        raise FileExistsError('For safety overwriting is not allowed', target)

    input_files = argv[1:]
    # count = 0
    # id_class = []
    # class_id = defaultdict(list)

    print('Computing class distribution')

    image_list = []
    label_list = []
    for input_filepath in input_files:
        with open(input_filepath, 'r') as infile:
            for line in infile:
                splits = line[:-1].split(',')
                if len(splits) == 2:
                    image_list.append(splits[0])
                    label_list.append(int(splits[1]))

    class_list = np.unique(label_list)
    # class_ind_map = {class_name:ind for ind, class_name in enumerate(class_list)}
    # ind_class_map = {ind:class_name for class_name, ind in class_ind_map.items()}


    labeled_image_list = []
    labeled_label_list = []
    unlabeled_image_list = []
    unlabeled_label_list = []

    for class_name in class_list:
        class_image_list = [fp for fp, l in zip(image_list, label_list) if l == class_name]
        
        class_label_image_list = list(np.random.choice(class_image_list, size=FLAGS.size, replace=False))
        # class_unlabeled_image_list = [fp for fp in class_image_list if fp not in class_label_image_list]
        class_unlabeled_image_list = class_image_list

        labeled_image_list += class_label_image_list
        labeled_label_list += [class_name for _ in class_label_image_list]
        unlabeled_image_list += class_unlabeled_image_list
        unlabeled_label_list += [class_name for _ in class_unlabeled_image_list]


    with open(target + '-labeled.txt', 'w') as outfile:
        for fp, l in zip(labeled_image_list, labeled_label_list):
            outfile.write(fp+','+str(l)+'\n')

    with open(target + '-unlabeled.txt', 'w') as outfile:
        for fp, l in zip(unlabeled_image_list, unlabeled_label_list):
            outfile.write(fp+','+str(l)+'\n')


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
