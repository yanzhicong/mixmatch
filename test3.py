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
"""MixMatch training.
- Ensure class consistency by producing a group of `nu` augmentations of the same image and guessing the label for the
  group.
- Sharpen the target distribution.
- Use the sharpened distribution directly as a smooth label in MixUp.
"""

import functools
import os

from absl import app
from absl import flags
from easydict import EasyDict
from libml import layers, utils, models
from libml.data_pair import DATASETS
from libml.layers import MixMode
import tensorflow as tf

FLAGS = flags.FLAGS


class MixMatch(models.MultiModel):

    def model(self, *args, **kwargs):

        return EasyDict()


def main(argv):
    del argv  # Unused.
    assert FLAGS.nu == 2
    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = MixMatch(
        './test_dir',
        dataset,
        nclass=dataset.nclass,
        arch=FLAGS.arch,
        scales=FLAGS.scales,
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)

    g = tf.Graph()
    with g.as_default():
        x_in = tf.placeholder(tf.float32, [None] + [84, 84, 3], 'x')
        y_in = tf.placeholder(tf.float32, [None] + [84, 84, 3], 'y')
        print(x_in.get_shape())

        with tf.name_scope('tes1'):
            ret = model.classifier(x_in,
                verbose=True,
                arch=FLAGS.arch,
                training=False,
                    scales=FLAGS.scales,
                    filters=FLAGS.filters,
                    repeat=FLAGS.repeat)
            print(ret.get_shape())

            ret = model.classifier(x_in,
                arch=FLAGS.arch,
                training=False,
                    scales=FLAGS.scales,
                    filters=FLAGS.filters,
                    repeat=FLAGS.repeat)
            print(ret.get_shape())

        with tf.name_scope('test2'):
            ret = model.classifier(x_in,
                arch=FLAGS.arch,
                training=False,
                    scales=FLAGS.scales,
                    filters=FLAGS.filters,
                    repeat=FLAGS.repeat)
            print(ret.get_shape())


            ret = model.cam_ext(x_in,
                arch=FLAGS.arch,
                training=False,
                    scales=FLAGS.scales,
                    filters=FLAGS.filters,
                    repeat=FLAGS.repeat)
            # print(ret[0].get_shape())

        # op = g.get_operation_by_name('Resnet18/batch_normalization/moving_mean/Initializer/zeros')
        # print(op)

        # op = g.get_operation_by_name('tes1/Cnn13/conv2d_7/Conv2D')
        # print(op)
        # for i in op.inputs:
            # print(i)


        model.print_weight_dict()

        # for node in g.as_graph_def().node:
        #     # print(node.name, node.device)
        #     print(node.name, node.op)

        writer = tf.summary.FileWriter(model.checkpoint_dir, graph=g)

        # op_list = 
    # model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10, step_summary_interval=100)
    # model.train(FLAGS.epochs, FLAGS.imgs_per_epoch // FLAGS.batch)
    # model.train(train_nimg=FLAGS.train_kimg << 10, report_nimg=FLAGS.report_kimg)


if __name__ == '__main__':
    utils.setup_tf()
    # flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    # flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    # flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    # flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 3, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 128, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 2, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset','miniimagenet.1@40-50')
    FLAGS.set_default('arch','resnet18')
    # FLAGS.set_default('batch', 64)
    # FLAGS.set_default('lr', 0.002)
    # FLAGS.set_default('lr_decay_rate', 0.001)
    # FLAGS.set_default('epochs', 100)
    # FLAGS.set_default('decay_start_epoch', 20)
    # FLAGS.set_default('imgs_per_epoch', 50000)
    app.run(main)
