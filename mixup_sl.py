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
"""mixup: Beyond Empirical Risk Minimization.

Adaption to SSL of MixUp: https://arxiv.org/abs/1710.09412
"""
import functools
import os

from absl import app
from absl import flags
from easydict import EasyDict
from libml import data, utils, models
from libml.vis import *

import tensorflow as tf

FLAGS = flags.FLAGS

from collections import OrderedDict
from mixup import Mixup


# @AutoVisDecorator

class MixupSL(Mixup):

    # def on_epoch_start(self, epoch_ind, epochs):
    #     super(MixupSL, self).on_epoch_start(epoch_ind, epochs)

    #     if epoch_ind % 5 != 0:
    #         return

    #     eval_dict = OrderedDict(
    #             train=self.dataset.labeled_data,
    #             valid=self.dataset.valid_data,
    #             test=self.dataset.test_data)

    #     for subset, data_source in eval_dict.items():

    #         nb_images = data_source.size
    #         choose_indices = np.random.choice(np.arange(nb_images), size=10, replace=False)
    #         labels = np.array([data_source.get_label(ind) for ind in choose_indices])
    #         images = np.array([data_source.get_img(ind) for ind in choose_indices])

    #         feats, cam = self.session.run(
    #             self.ops.cam_op,
    #             feed_dict={
    #                 self.ops.x: images.astype(np.float32) / 255.0
    #             })

    #         data_list = [
    #             ImageVSTensorData('class activation map', images, cam, point_out_ind={
    #                 'l' : labels,
    #                 'p' : np.argmax(feats.logits, axis=1),
    #             }, channel_wise_normalize=False),
    #         ]
    #         watch_out_list = {
    #             'cnn13' : ['bn2_3', 'bn1_2'],
    #             'resnet18' : ['res1b', 'res2b', 'res3b']
    #         }[FLAGS.arch]
    #         data_list += [ImageVSTensorData(l, images, feats[l]) for l in watch_out_list]

    #         draw_data_list_to_html(data_list, os.path.join(self.train_dir, 'cam_view_'+subset), epoch_ind=int(self.session.run(self.epoch)))


    # def on_epoch_end(self, epoch_ind, epochs):
    #     super(MixupSL, self).on_epoch_end(epoch_ind, epochs)


    def model(self, lr, wd, ema, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]

        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        classifier = functools.partial(self.classifier, verbose=True, **kwargs)

        def get_logits(x):
            logits = classifier(x, training=True)
            return logits

        x, labels_x = self.augment(x_in, tf.one_hot(l_in, self.nclass), **kwargs)
        logits_x = get_logits(x)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(x_in, training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        self.train_step_monitor_summary = tf.summary.merge([
            tf.summary.scalar('losses/xe', loss_xe)
        ])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, training=False)),
            # cam_op=self.cam_ext(x_in, training=False, **kwargs)
            )


def main(argv):
    del argv  # Unused.
    dataset = data.DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = MixupSL(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        beta=FLAGS.beta,
        epochs=FLAGS.epochs,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)

    # train model in supervised setting
    model.train(FLAGS.epochs, FLAGS.imgs_per_epoch // FLAGS.batch, ssl=False)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset', 'cifar10.3@4000-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('lr_decay_rate', 0.001)
    FLAGS.set_default('epochs', 100)
    FLAGS.set_default('decay_start_epoch', 20)
    FLAGS.set_default('imgs_per_epoch', 50000)
    app.run(main)

