

import functools
import os
import numpy as np
from absl import app
from absl import flags
from easydict import EasyDict
from libml import layers, utils, models
from libml.data import DATASETS
from libml.layers import MixMode
from glob import glob

import tensorflow as tf
from tensorflow.python.client import device_lib


FLAGS = flags.FLAGS


_GPUS = None

def get_available_gpus():
    global _GPUS
    if _GPUS is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        local_device_protos = device_lib.list_local_devices(session_config=config)
        _GPUS = tuple([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return _GPUS



def get_config():
    config = tf.ConfigProto()
    if len(get_available_gpus()) > 1:
        config.allow_soft_placement = True
    if FLAGS.log_device_placement:
        config.log_device_placement = True
    config.gpu_options.allow_growth = True
    return config





def main(argv):
    del argv  # Unused.
    # assert FLAGS.nu == 2
    dataset = DATASETS[FLAGS.dataset](use_pseudo_label=True)
    batch = 64


    def get_dataset_reader(dataset, batch, prefetch=16):
        d = dataset.batch(batch).prefetch(
            prefetch).make_initializable_iterator()
        return d.initializer, d.get_next()

    eval_labeled = get_dataset_reader(dataset.eval_labeled, batch)
    valid = get_dataset_reader(dataset.valid, batch)
    test = get_dataset_reader(dataset.test, batch)


    subset_list = ['train_labeled', 'valid', 'test']
    dataset_list = [eval_labeled, valid, test]

    # with mutex:

    def count_label(labels):
        label_set = np.unique(labels)
        for l in label_set:
            print('\t', l, (labels==l).sum())


    with tf.Session(config=get_config()) as sess:
        
        for subset, dataset in zip(subset_list, dataset_list):
            labels = []

            sess.run([dataset[0]])
            while True:
                try:
                    v = sess.run(dataset[1])
                    labels.append(v['label'])
                except tf.errors.OutOfRangeError:
                    break
            labels = np.concatenate(labels, axis=0)

            print(subset)
            count_label(labels)
    

if __name__ == '__main__':
    utils.setup_tf()
    # flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    # flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    # flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    # flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    # flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    # flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    # flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    # FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('dataset', 'miniimagenet.3@40-50')
    # FLAGS.set_default('batch', 64)
    # FLAGS.set_default('lr', 0.002)
    # FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)




# if __name__ == '__main__':
#     for key, value in DATASETS.items():
#         print(key)
