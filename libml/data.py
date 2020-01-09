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
from easydict import EasyDict

from libml import utils

_DATA_CACHE = None
DATA_DIR = os.environ['ML_DATA']
flags.DEFINE_string('dataset', 'cifar10.1@4000-5000', 'Data to train on.')
flags.DEFINE_integer('para_parse', 4, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')
flags.DEFINE_integer('num_pseudo_label_channels', 2, 'Size of dataset shuffling.')
flags.DEFINE_string('p_unlabeled', '', 'Probability distribution of unlabeled.')
flags.DEFINE_bool('whiten', False, 'Whether to normalize images.')
FLAGS = flags.FLAGS



def record_parse(dataset: tf.data.Dataset) -> tf.data.Dataset:
    def record_parse_internal(serialized_example):
        return tf.parse_single_example(
            serialized_example,
            features={'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)})
    return dataset.map(record_parse_internal)

def random_split_ind(label_list, num_valid_per_class):
    label_set = np.unique(label_list)
    valid_ind_list = []
    for l in label_set:
        valid_ind_list += list(np.random.choice(np.where(label_list==l)[0], size=num_valid_per_class, replace=False))
    valid_ind_list.sort()
    return valid_ind_list


def extract_image_and_label_list(filenames):
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
    return EasyDict({
        'images':image_list, 
        'labels':label_list
    })


def extract_tf_dataset(dataset):
    data = []
    with tf.Session(config=utils.get_config()) as session:
        dataset = dataset.prefetch(16)
        it = dataset.make_one_shot_iterator().get_next()
        try:
            while 1:
                data.append(session.run(it))
        except tf.errors.OutOfRangeError:
            pass
    images = [x['image'] for x in data]
    labels = [x['label'] for x in data]
    return EasyDict({
        'images': images,
        'labels': labels,
    })


def split_data_by_ind(data_dict, ind_list):

    inc_data = EasyDict()
    exc_data = EasyDict()
    
    def ind_include_filter(data_list, ind_list):
        ind_list.sort()
        return [data_list[ind] for ind in ind_list]
        
    def ind_exclude_filter(data_list, ind_list):
        ind_list = list(set(np.arange(len(data_list))) - set(ind_list))
        ind_list.sort()
        return [data_list[ind] for ind in ind_list]

    for field in ['images', 'labels']:
        inc_data[field] = ind_include_filter(data_dict[field], ind_list)
        exc_data[field] = ind_exclude_filter(data_dict[field], ind_list)

    return inc_data, exc_data


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


class DataSource:
    def __init__(self, filename=None):
        self.datasource=None
        self.load_data(filename)

    def load_data(self, filename):
        self.filename = filename
        if filename is not None:
            print(filename)
            assert self.filename.split('.')[-1] in ['tfrecord', 'txt']
            if self.filename.endswith('tfrecord'):
                self.load_tf_record_data_source()
            elif self.filename.endswith('txt'):
                self.load_txt_data_source()

    @property
    def size(self):
        return len(self.datasource['images'])

    @property
    def labels(self):
        return self.datasource['labels']

    @property
    def data_type(self):
        return self.filename.split('.')[-1]

    def get_img(self, ind):
        if self.data_type == 'txt':
            filein = np.fromfile(self.datasource['images'][ind].decode('utf-8'), dtype=np.uint8)
            return cv2.imdecode(filein, cv2.IMREAD_COLOR)

    def get_label(self, ind):
        return self.datasource['labels'][ind]

    def get_cls(self, ind):
        return str(self.datasource['labels'][ind])

    def load_tf_record_data_source(self):
        def dataset(filenames: list) -> tf.data.Dataset:
            filenames = sorted(sum([glob.glob(x) for x in filenames], []))
            if not filenames:
                raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
            return tf.data.TFRecordDataset(filenames)
        self.datasource = extract_tf_dataset(record_parse(dataset([self.filename])))


    def load_txt_data_source(self): 
        self.datasource = extract_image_and_label_list([self.filename])

    def split_by_indices(self, indices):
        a = DataSource()
        b = DataSource()
        a.filename = self.filename
        b.filename = self.filename
        a.datasource, b.datasource = split_data_by_ind(self.datasource, indices)
        return a, b

    @classmethod
    def concatenate(cls, a, b):
        ret = DataSource()
        ret.filename = a.filename + '_' + b.filename
        ret.datasource = dict(
            images = a.datasource.images + b.datasource.images,
            labels = a.datasource.labels + b.datasource.labels,
        )
        return ret
    
    # deprecated
    def create_pseudo_labels(self):
        if 'pseudo_labels' not in self.datasource:
            self.datasource['pseudo_labels'] = tf.get_variable('pseudo_labels', shape=[self.size, FLAGS.num_pseudo_label_channels], dtype=tf.float32, initializer=tf.zeros_initializer())

    def create_tf_dataset(self, num_parallel=1):
        def cv2_imread(filepath):
            filein = np.fromfile(filepath.decode('utf-8'), dtype=np.uint8)
            cv_img = cv2.imdecode(filein, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            return cv_img

        def to_dataset(img_proc_func, num_parallel=1):
            data_list = [np.arange(self.size, dtype=np.int64), self.datasource['images'], np.array(self.datasource['labels']).astype(np.int64)]
            if 'pseudo_labels' in self.datasource:
                data_list.append(self.datasource['pseudo_labels'])
            dataset = tf.data.Dataset.from_tensor_slices(tuple(data_list))

            if 'pseudo_labels' in self.datasource:
                dataset = dataset.map(map_func=lambda ind, img, label, plabel: 
                                            {'index' : ind,
                                            'image' : img_proc_func(img), 
                                            'label' : label,
                                            'pseudo_label' : plabel}, num_parallel_calls=num_parallel)
            else:
                dataset = dataset.map(map_func=lambda ind, img, label: 
                                            {'index' : ind,
                                            'image' : img_proc_func(img), 
                                            'label' : label}, num_parallel_calls=num_parallel)
            return dataset

        if self.data_type == 'txt':
            return to_dataset(lambda x : tf.py_func(cv2_imread, inp=[x], Tout=tf.float32), num_parallel=num_parallel)
        elif self.data_type == 'tfrecord':
            return to_dataset(lambda x : tf.cast(tf.image.decode_image(x), tf.float32) / 255.0 , num_parallel=num_parallel)




class DataSet(object):

    def __init__(self, name, seed, label, valid, full_dataset=False):
        
        fullname = '.%d@%d' % (seed, label)
        self.root = os.path.join(DATA_DIR, 'SSL', name + fullname)
        self.pseudo_labels = None

        if full_dataset:
            self.labeled_data_path = os.path.join(DATA_DIR, '%s-train.tfrecord' % name)
            self.unlabeled_data_path = os.path.join(DATA_DIR, '%s-train.tfrecord' % name)
        else:
            self.labeled_data_path = self.root + '-label.tfrecord'
            self.unlabeled_data_path = self.root + '-unlabel.tfrecord'
        self.test_data_path = os.path.join(DATA_DIR, '%s-test.tfrecord' % name)

    def assign_plabel(self, sess, plabel):
        if self.pseudo_labels is not None:
            sess.run(self.pseudo_labels.assign(plabel))


    @classmethod
    def creator(cls, name, seed, label, valid, augment, do_memoize=True, colors=3,
                nclass=10, height=32, width=32, name_suffix='', full_dataset=False):

        if not isinstance(augment, list):
            augment = [augment] * 2

        fn = memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)
        fullname = '.%d@%d' % (seed, label)
        dataset_name = name + name_suffix + fullname + '-' + str(valid) if not full_dataset else name + name_suffix + '-' + str(valid)
        
        # use_pseudo_label : deprecated arg
        def create(split_indices=None, use_pseudo_label=False):
            p_labeled = p_unlabeled = None
            para1 = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
            para2 = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse

            instance = cls(name, seed, label, valid, full_dataset=full_dataset)

            if FLAGS.p_unlabeled:
                sequence = FLAGS.p_unlabeled.split(',')
                p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
                p_unlabeled /= np.max(p_unlabeled)

            with tf.name_scope('DatasetCreator'):
    
                labeled_data, unlabeled_data, test_data = map(lambda x:DataSource(x), [instance.labeled_data_path, instance.unlabeled_data_path, 
                            instance.test_data_path])

                if split_indices is None:
                    
                    if unlabeled_data.data_type == 'tfrecord':      # just patch
                        split_indices = random_split_ind(unlabeled_data.labels, num_valid_per_class=valid//nclass)
                    else:
                        split_indices = random_split_ind(unlabeled_data.labels, num_valid_per_class=valid)


                # 从unlabeled_data中分离出valid_data
                valid_data, unlabeled_data = unlabeled_data.split_by_indices(split_indices)


                if full_dataset:
                    # 使用full_dataset时，labeled_data与unlabeled_data相同，将valid_data从labeled_data中分离出来
                    _, labeled_data = labeled_data.split_by_indices(split_indices)

                if use_pseudo_label:
                    unlabeled_data.create_pseudo_labels()

                # data source attributes
                instance.__dict__.update(dict(
                    labeled_data=labeled_data,
                    unlabeled_data=unlabeled_data,
                    train_data=DataSource.concatenate(labeled_data, unlabeled_data),
                    valid_data=valid_data,
                    test_data=test_data
                ))

                # build tf dataset pipeline
                train_labeled, train_unlabeled, eval_labeled, eval_unlabeled, eval_valid, test = map(lambda x:x.create_tf_dataset(para2), [labeled_data, unlabeled_data] * 2 + [valid_data, test_data])

                train_labeled = fn(train_labeled).map(augment[0], para1)
                train_unlabeled = fn(train_unlabeled).map(augment[1], para1)

                if FLAGS.whiten:
                    if unlabeled_data.data_type == 'tfrecord':      # just patch
                        mean, std = compute_mean_std(eval_unlabeled)
                    else:
                        mean_var = json.loads(open(os.path.join(DATA_DIR, '%s-meanvar.json' % name), 'r').read())
                        mean = np.array(mean_var['mean'])
                        std = np.array(mean_var['var'])
                else:
                    mean, std = 0, 1

                instance.__dict__.update(dict(
                    name=dataset_name,
                    train_labeled=train_labeled,
                    train_unlabeled=train_unlabeled,
                    eval_labeled=eval_labeled,
                    eval_unlabeled=eval_unlabeled,
                    valid=eval_valid,
                    test=test,
                    nclass=nclass, colors=colors, p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                    height=height, width=width, mean=mean, std=std
                ))

                if use_pseudo_label:
                    instance.__dict__.update(dict(
                        pseudo_labels=unlabeled_data.datasource['pseudo_labels'],
                    ))

                return instance

        return dataset_name, create


class TxtDataSet(DataSet):

    def __init__(self, name, seed, label, valid, full_dataset=False):
        
        fullname = '.%d@%d' % (seed, label)
        root = os.path.join(DATA_DIR, 'SSL', name + fullname)

        if full_dataset:
            self.labeled_data_path = os.path.join(DATA_DIR, '%s-train.txt' % name)
            self.unlabeled_data_path = os.path.join(DATA_DIR, '%s-train.txt' % name)
        else:
            self.labeled_data_path = root + '-labeled.txt'
            self.unlabeled_data_path = root + '-unlabeled.txt'
        self.test_data_path = os.path.join(DATA_DIR, '%s-test.txt' % name)


augment_stl10 = lambda x: {k:v if k != 'image' else augment_shift(augment_mirror(x['image']), 12) for k, v in x.items()}
augment_cifar10 = lambda x: {k:v if k != 'image' else augment_shift(augment_mirror(x['image']), 4) for k, v in x.items()}
augment_svhn = lambda x: {k:v if k != 'image' else augment_shift(x['image'], 4) for k, v in x.items()}
argument_miniimagenet = lambda x: {k:v if k != 'image' else augment_shift(augment_mirror(x['image']), 4) for k, v in x.items()}


DATASETS = {}
DATASETS.update([DataSet.creator('cifar10', seed, label, valid, augment_cifar10, do_memoize=False)
                    for seed, label, valid in
                        itertools.product(range(6), [250, 500, 1000, 2000, 4000, 8000], [1, 5000])])

DATASETS.update([DataSet.creator('cifar100', seed, label, valid, augment_cifar10, nclass=100, do_memoize=False)
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

DATASETS.update([TxtDataSet.creator('miniimagenet', 0, 0, valid, argument_miniimagenet, width=84, height=84, nclass=100, do_memoize=False, full_dataset=True)
                    for valid in [1, 50]])


