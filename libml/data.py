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
"""Input data for image models.
"""

import glob
import itertools
import os

import numpy as np
import tensorflow as tf
from absl import flags
from tqdm import tqdm
import numpy as np
import json
import cv2

from libml import utils

_DATA_CACHE = None
DATA_DIR = os.environ['ML_DATA']
flags.DEFINE_string('dataset', 'cifar10.1@4000-5000', 'Data to train on.')
flags.DEFINE_integer('para_parse', 4, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')
flags.DEFINE_string('p_unlabeled', '', 'Probability distribution of unlabeled.')
flags.DEFINE_bool('whiten', False, 'Whether to normalize images.')
FLAGS = flags.FLAGS


def record_parse(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    label = features['label']
    return dict(image=image, label=label)


def default_parse(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:
    para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse
    return dataset.map(parse_fn, num_parallel_calls=para)


# def default_parse2(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:
#     para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse
#     return dataset.map(parse_fn, num_parallel_calls=para)


def dataset(filenames: list) -> tf.data.Dataset:
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    return tf.data.TFRecordDataset(filenames)


def txt_dataset_imglist_w_label(filenames: list, split_valid:int=0) -> tf.data.Dataset:
    """
        from txt file to tf.data.Dataset
        if split_valid == 0, return only one dataset, otherwise will return dataset and valid_dataset
    """

    print(filenames)
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did not find any txt file')

    image_list = []
    label_list = []
    for filename in filenames:
        with open(filename, 'r') as infile:
            for line in infile:
                splits = line[:-1].split(',')
                if len(splits) == 2:
                    image_list.append(splits[0].encode('utf-8'))
                    label_list.append(int(splits[1]))

    def cv2_imread(filepath):
        filein = np.fromfile(filepath.decode('utf-8'), dtype=np.uint8)
        cv_img = cv2.imdecode(filein, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        return cv_img

    para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse

    def list_to_dataset(i_list, l_list, num_parallel_calls):
        dataset = tf.data.Dataset.from_tensor_slices((i_list, l_list))
        dataset = dataset.map(map_func=lambda img_filepath, label : 
                                    {'image' : tf.py_func(func=cv2_imread, inp=[img_filepath,], Tout=tf.float32), 
                                    'label' : label}, num_parallel_calls=num_parallel_calls)
        return dataset

    def ind_filter(item_list, ind_list):
        ind_list.sort()
        return [item_list[ind] for ind in ind_list]
        
    def ind_exclude_filter(item_list, ind_list):
        ind_list = list(set(np.arange(len(item_list))) - set(ind_list))
        ind_list.sort()
        return [item_list[ind] for ind in ind_list]
        
    if split_valid == 0:
        return list_to_dataset(image_list, label_list, para)
    else:
        label_set = np.unique(label_list)
        valid_ind_list = []
        for l in label_set:
            valid_ind_list += list(np.random.choice(np.where(label_list==l)[0], size=split_valid, replace=False))
        
        return list_to_dataset(ind_exclude_filter(image_list, valid_ind_list), ind_exclude_filter(label_list, valid_ind_list), para),          \
               list_to_dataset(ind_filter(image_list, valid_ind_list), ind_filter(label_list, valid_ind_list), para)



# def txt_dataset_imglist_wo_label(filenames: list) -> tf.data.Dataset:
#     filenames = sorted(sum([glob.glob(x) for x in filenames], []))
#     if not filenames:
#         raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
#     image_list = []
#     for filename in filenames:
#         with open(filename, 'r') as infile:
#             for line in infile:
#                 splits = line[:-1].split(',')
#                 if len(splits) == 2:
#                     image_list.append(splits[0].encode('utf-8'))
#                     label_list.append(int(splits[1]))
#     def cv2_imread(filepath):
#         filein = np.fromfile(filepath, dtype=np.uint8)
#         cv_img = cv2.imdecode(filein, cv2.IMREAD_COLOR)
#         return cv_img
#     dataset = tf.data.Dataset.from_tensor_slices(image_list)
#     dataset2 = tf.data.Dataset.from_tensor_slices(image_list)

#     dataset = dataset.map(map_func=l)
#     dataset = tf.data.Dataset.zip([dataset, dataset2])



def memoize(dataset: tf.data.Dataset) -> tf.data.Dataset:
    data = []
    with tf.Session(config=utils.get_config()) as session:
        dataset = dataset.prefetch(16)
        it = dataset.make_one_shot_iterator().get_next()
        try:
            while 1:
                data.append(session.run(it))
        except tf.errors.OutOfRangeError:
            pass
    images = np.stack([x['image'] for x in data])
    labels = np.stack([x['label'] for x in data])

    def tf_get(index):
        def get(index):
            return images[index], labels[index]

        image, label = tf.py_func(get, [index], [tf.float32, tf.int64])
        return dict(image=image, label=label)

    dataset = tf.data.Dataset.range(len(data)).repeat()
    dataset = dataset.shuffle(len(data) if len(data) < FLAGS.shuffle else FLAGS.shuffle)
    return dataset.map(tf_get)


def augment_mirror(x):
    return tf.image.random_flip_left_right(x)


def augment_shift(x, w):
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.random_crop(y, tf.shape(x))

def augment_noise(x, std):
    return x + std * tf.random_normal(tf.shape(x), dtype=x.dtype)


def compute_mean_std(data: tf.data.Dataset):
    data = data.map(lambda x: x['image']).batch(1024).prefetch(1)
    data = data.make_one_shot_iterator().get_next()
    count = 0
    stats = []
    with tf.Session(config=utils.get_config()) as sess:
        def iterator():
            while True:
                try:
                    yield sess.run(data)
                except tf.errors.OutOfRangeError:
                    break

        for batch in tqdm(iterator(), unit='kimg', desc='Computing dataset mean and std'):
            ratio = batch.shape[0] / 1024.
            count += ratio
            stats.append((batch.mean((0, 1, 2)) * ratio, (batch ** 2).mean((0, 1, 2)) * ratio))
    mean = sum(x[0] for x in stats) / count
    sigma = sum(x[1] for x in stats) / count - mean ** 2
    std = np.sqrt(sigma)
    print('Mean %s  Std: %s' % (mean, std))
    return mean, std


class DataSet:
    def __init__(self, name, train_labeled, train_unlabeled, test, valid, eval_labeled, eval_unlabeled,
                 height=32, width=32, colors=3, nclass=10, mean=0, std=1, p_labeled=None, p_unlabeled=None):
        self.name = name
        self.train_labeled = train_labeled
        self.train_unlabeled = train_unlabeled
        self.eval_labeled = eval_labeled
        self.eval_unlabeled = eval_unlabeled
        self.test = test
        self.valid = valid
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass
        self.mean = mean
        self.std = std
        self.p_labeled = p_labeled
        self.p_unlabeled = p_unlabeled

    @classmethod
    def creator(cls, name, seed, label, valid, augment, parse_fn=default_parse, do_memoize=True, colors=3,
                nclass=10, height=32, width=32, name_suffix=''):
        if not isinstance(augment, list):
            augment = [augment] * 2
        fullname = '.%d@%d' % (seed, label)
        root = os.path.join(DATA_DIR, 'SSL', name + fullname)
        fn = memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)

        def create():
            p_labeled = p_unlabeled = None
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment

            if FLAGS.p_unlabeled:
                sequence = FLAGS.p_unlabeled.split(',')
                p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
                p_unlabeled /= np.max(p_unlabeled)

            train_labeled = parse_fn(dataset([root + '-label.tfrecord']))
            train_unlabeled = parse_fn(dataset([root + '-unlabel.tfrecord']).skip(valid))
            if FLAGS.whiten:
                mean, std = compute_mean_std(train_labeled.concatenate(train_unlabeled))
            else:
                mean, std = 0, 1

            return cls(name + name_suffix + fullname + '-' + str(valid),
                       train_labeled=fn(train_labeled).map(augment[0], para),
                       train_unlabeled=fn(train_unlabeled).map(augment[1], para),
                       eval_labeled=parse_fn(dataset([root + '-label.tfrecord'])),
                       eval_unlabeled=parse_fn(dataset([root + '-unlabel.tfrecord']).skip(valid)),
                       valid=parse_fn(dataset([root + '-unlabel.tfrecord']).take(valid)),
                       test=parse_fn(dataset([os.path.join(DATA_DIR, '%s-test.tfrecord' % name)])),
                       nclass=nclass, colors=colors, p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                       height=height, width=width, mean=mean, std=std)

        return name + name_suffix + fullname + '-' + str(valid), create



class TxtDataSet:
    def __init__(self, name, train_labeled, train_unlabeled, test, valid, eval_labeled, eval_unlabeled,
                 height=32, width=32, colors=3, nclass=10, mean=0, std=1, p_labeled=None, p_unlabeled=None):
        self.name = name
        self.train_labeled = train_labeled
        self.train_unlabeled = train_unlabeled
        self.eval_labeled = eval_labeled
        self.eval_unlabeled = eval_unlabeled
        self.test = test
        self.valid = valid
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass
        self.mean = mean
        self.std = std
        self.p_labeled = p_labeled
        self.p_unlabeled = p_unlabeled

    @classmethod
    def creator(cls, name, seed, label, valid, augment, parse_fn=default_parse, do_memoize=True, colors=3,
                nclass=10, height=32, width=32, name_suffix=''):
        if not isinstance(augment, list):
            augment = [augment] * 2
        fullname = '.%d@%d' % (seed, label)
        root = os.path.join(DATA_DIR, 'SSL', name + fullname)
        fn = memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)

        def create():
            p_labeled = p_unlabeled = None
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment

            if FLAGS.p_unlabeled:
                sequence = FLAGS.p_unlabeled.split(',')
                p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
                p_unlabeled /= np.max(p_unlabeled)

            train_labeled = txt_dataset_imglist_w_label([root + '-labeled.txt'])
            train_unlabeled, train_valid = txt_dataset_imglist_w_label([root + '-unlabeled.txt'], split_valid=valid)

            eval_labeled = txt_dataset_imglist_w_label([root + '-labeled.txt'])
            eval_unlabeled, eval_valid = txt_dataset_imglist_w_label([root + '-unlabeled.txt'], split_valid=valid)

            test = txt_dataset_imglist_w_label([os.path.join(DATA_DIR, '%s-test.txt' % name)])


            if FLAGS.whiten:
                mean_var = json.loads(open(os.path.join(DATA_DIR, '%s-meanvar.json' % name), 'r').read())
                mean = np.array(mean_var['mean'])
                std = np.array(mean_var['var'])
            else:
                mean, std = 0, 1

            return cls(name + name_suffix + fullname + '-' + str(valid),
                       train_labeled=fn(train_labeled).map(augment[0], para),
                       train_unlabeled=fn(train_unlabeled).map(augment[1], para),
                       eval_labeled=eval_labeled,
                       eval_unlabeled=eval_unlabeled,
                       valid=eval_valid,
                       test=test,
                       nclass=nclass, colors=colors, p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                       height=height, width=width, mean=mean, std=std)

        return name + name_suffix + fullname + '-' + str(valid), create



    @classmethod
    def full_creator(cls, name, valid, augment, parse_fn=default_parse, do_memoize=True, colors=3,
                nclass=10, height=32, width=32, name_suffix=''):

        if not isinstance(augment, list):
            augment = [augment] * 2

        fn = memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)

        def create():
            p_labeled = p_unlabeled = None
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment

            if FLAGS.p_unlabeled:
                sequence = FLAGS.p_unlabeled.split(',')
                p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
                p_unlabeled /= np.max(p_unlabeled)

            train_labeled = txt_dataset_imglist_w_label([os.path.join(DATA_DIR, '%s-train.txt' % name)])
            train_unlabeled, train_valid = txt_dataset_imglist_w_label([os.path.join(DATA_DIR, '%s-train.txt' % name)], split_valid=valid)

            eval_labeled = txt_dataset_imglist_w_label([os.path.join(DATA_DIR, '%s-train.txt' % name)])
            eval_unlabeled, eval_valid = txt_dataset_imglist_w_label([os.path.join(DATA_DIR, '%s-train.txt' % name)], split_valid=valid)

            test = txt_dataset_imglist_w_label([os.path.join(DATA_DIR, '%s-test.txt' % name)])


            if FLAGS.whiten:
                mean_var = json.loads(open(os.path.join(DATA_DIR, '%s-meanvar.json' % name), 'r').read())
                mean = np.array(mean_var['mean'])
                std = np.array(mean_var['var'])
            else:
                mean, std = 0, 1

            return cls(name + name_suffix,
                       train_labeled=fn(train_labeled).map(augment[0], para),
                       train_unlabeled=fn(train_unlabeled).map(augment[1], para),
                       eval_labeled=eval_labeled,
                       eval_unlabeled=eval_unlabeled,
                       valid=eval_valid,
                       test=test,
                       nclass=nclass, colors=colors, p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                       height=height, width=width, mean=mean, std=std)

        return name + name_suffix + '-' + str(valid), create


augment_stl10 = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 12), label=x['label'])
augment_cifar10 = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 4), label=x['label'])
augment_svhn = lambda x: dict(image=augment_shift(x['image'], 4), label=x['label'])
argument_miniimagenet = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 4), label=x['label'])


DATASETS = {}
DATASETS.update([DataSet.creator('cifar10', seed, label, valid, augment_cifar10)
                    for seed, label, valid in
                        itertools.product(range(6), [250, 500, 1000, 2000, 4000, 8000], [1, 5000])])

DATASETS.update([DataSet.creator('cifar100', seed, label, valid, augment_cifar10, nclass=100)
                    for seed, label, valid in
                        itertools.product(range(6), [10000], [1, 5000])])

DATASETS.update([DataSet.creator('stl10', seed, label, valid, augment_stl10, height=96, width=96, do_memoize=False)
                    for seed, label, valid in
                        itertools.product(range(6), [1000, 5000], [1, 500])])

DATASETS.update([DataSet.creator('svhn', seed, label, valid, augment_svhn, do_memoize=False)
                    for seed, label, valid in
                        itertools.product(range(6), [250, 500, 1000, 2000, 4000, 8000], [1, 5000])])

DATASETS.update([DataSet.creator('svhn_noextra', seed, label, valid, augment_svhn, do_memoize=False)
                    for seed, label, valid in
                        itertools.product(range(6), [250, 500, 1000, 2000, 4000, 8000], [1, 5000])])

DATASETS.update([TxtDataSet.creator('miniimagenet', seed, label, valid, argument_miniimagenet, width=84, height=84, nclass=100, do_memoize=False)
                    for seed, label, valid in
                        itertools.product(range(6), [40, 100], [1, 50])])

DATASETS.update([TxtDataSet.full_creator('miniimagenet', valid, argument_miniimagenet, width=84, height=84, nclass=100, do_memoize=False)
                    for valid in [1, 50]])
