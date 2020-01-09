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
from libml.data import DATASETS, DataSource
from libml.layers import MixMode
from libml.extern import ClassifySemiWithPLabel
from libml.vis import *
import tensorflow as tf
import numpy as np

FLAGS = flags.FLAGS

from mixup import Mixup

from third_party.lp_utils import graph_laplace

# import pickle as pkl

class DeepLP(ClassifySemiWithPLabel):

    def augment(self, x, l, w=None, beta=1.0, **kwargs):
        del kwargs

        if beta < 1e-5:
            return x, l
        else:
            mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1, 1])
            mix = tf.maximum(mix, 1 - mix)
            xmix = x * mix + x[::-1] * (1 - mix)
            lmix = l * mix[:, :, 0, 0] + l[::-1] * (1 - mix[:, :, 0, 0])
            if w is None:
                return xmix, lmix
            else:
                wmix = w * mix[:, 0, 0, 0] + w[::-1] * (1-mix[:, 0, 0, 0])
                return xmix, lmix, wmix


    def update_pseudo_label(self):

        subset_list = ['labeled', 'unlabeled']
        dataset_list = [self.eval_labeled, self.eval_unlabeled]

        feats_dict = {}
        ema_feats_dict = {}
        labels_dict = {}

        for subset, dataset in zip(subset_list, dataset_list):
            self.session.run(dataset[0])

            feats = []
            ema_feats = []
            labels = []
        
            while True:
                try:
                    v = self.session.run(dataset[1])
                    f1, f2 = self.session.run(
                        [self.ops.feat_ext_raw, self.ops.feat_ext_op],
                        feed_dict={
                            self.ops.x: v['image'],
                        })
                    feats.append(f1)
                    ema_feats.append(f2)
                    labels.append(v['label'])
                    
                except tf.errors.OutOfRangeError:
                    break

            feats_dict[subset] = np.concatenate(feats, axis=0)
            ema_feats_dict[subset] = np.concatenate(ema_feats, axis=0)
            labels_dict[subset] = np.concatenate(labels, axis=0)

        num_labeled = len(feats_dict['labeled'])
        num_unlabeled = len(feats_dict['unlabeled'])

        label = labels_dict['labeled']
        label_indices = np.arange(len(label))

        def _get(d, l):
            return np.concatenate([d[i] for i in l], axis=0)

        all_feats = _get(feats_dict, subset_list)
        all_ema_feats = _get(ema_feats_dict, subset_list)
    
        data = FeatureSpaceRecordData2('network', all_feats, self.dataset.train_data, self.dataset.labeled_data.size)
        data2 = FeatureSpaceRecordData2('ema network', all_ema_feats, self.dataset.train_data, self.dataset.labeled_data.size)
        draw_data_list_to_html([data, data2], os.path.join(self.train_dir, 'feature_space'), int(self.session.run(self.epoch)))

        # label propagation on feature space from classifer
        Y0, W0 = graph_laplace(all_feats, label, label_indices, self.nclass)
        # label propagation on feature space from ema classifier
        Y1, W1 = graph_laplace(all_ema_feats, label, label_indices, self.nclass)

        Y0, W0, Y1, W1 = map(lambda x:x[num_labeled:], [Y0, W0, Y1, W1])

        self.session.run(self.ops.pseudo_labels.assign(np.array([Y0, W0, Y1, W1, labels_dict['unlabeled']], dtype=np.float32).transpose()))


    def add_summaries(self, feed_extra=None, **kwargs):
        super(DeepLP, self).add_summaries(feed_extra=feed_extra, **kwargs)

        def gen_stats():
            plabel = self.session.run(self.ops.pseudo_labels)
            acc1, wacc1 = self.eval_pesudo_label_acc(plabel[:, 0].astype(np.int32), plabel[:, 1], self.dataset.unlabeled_data.labels)
            acc2, wacc2 = self.eval_pesudo_label_acc(plabel[:, 2].astype(np.int32), plabel[:, 3], self.dataset.unlabeled_data.labels)

            epoch_ind = int(self.session.run(self.epoch))
            self.plotter.scalar('plabel_acc', epoch_ind, {
                'accuracy/plabel' : acc1,
                'weighted_accuracy/plabel' : wacc1,
                'accuracy/ema_plabel' : acc2,
                'weighted_accuracy/ema_plabel' : wacc2,
            })
            return np.array([acc1, wacc1, acc2, wacc2], dtype=np.float32)

        accuracies = tf.py_func(gen_stats, [], tf.float32)

        self.summary_list += [
            tf.summary.scalar('accuracy/plabel', accuracies[0]),
            tf.summary.scalar('weighted_accuracy/plabel', accuracies[1]),
            tf.summary.scalar('accuracy/ema_plabel', accuracies[2]),
            tf.summary.scalar('weighted_accuracy/ema_plabel', accuracies[3]),
        ]


    def model(self, batch, lr, wd, ema, beta, w_match, warmup_kimg=1024, mixmode='xxy.yxy', **kwargs):

        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        y_in_ind = tf.placeholder(tf.int64, [None], 'y_ind')

        pseudo_labels = tf.get_variable('pseudo_labels', dtype=tf.float32, shape=[self.dataset.unlabeled_data.size, 5], initializer=tf.zeros_initializer())
        plabel_in = tf.stop_gradient(tf.nn.embedding_lookup(pseudo_labels, y_in_ind))

        wd *= lr
        w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)

        classifier = functools.partial(self.classifier, **kwargs)
        feat_ext = functools.partial(self.feature_ext, **kwargs)

        lx = tf.one_hot(l_in, self.nclass)
        ly_l = tf.one_hot(tf.cast(plabel_in[ :, 2], tf.int64), self.nclass)
        ly_w = tf.cast(plabel_in[:, 3], tf.float32)

        x, lx = self.augment(x_in, lx, beta=beta)
        y, ly, wy = self.augment(y_in, ly_l, ly_w)

        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        xlogits = classifier(x, training=True)
        ylogits = classifier(y, training=True)

        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=lx, logits=xlogits)
        loss_xe = tf.reduce_mean(loss_xe)

        loss_xeu = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ly, logits=ylogits)
        loss_xeu = tf.reduce_mean(loss_xeu * wy)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_xeu, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(x_in, training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        self.train_step_monitor_summary = tf.summary.merge([
            tf.summary.scalar('losses/xe', loss_xe),
            tf.summary.scalar('losses/xeu', loss_xeu),
            tf.summary.scalar('vars/warm_up_weight', w_match),
            tf.summary.scalar('vars/learning_rate', lr),
            tf.summary.histogram('vars/ly_w', ly_w),
            tf.summary.histogram('vars/wy', wy),
            ])

        return EasyDict(
            x=x_in,
            y=y_in,
            y_ind=y_in_ind,
            label=l_in,
            pseudo_labels=pseudo_labels,
            train_op=train_op,
            tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            feat_ext_raw=feat_ext(x_in, training=False),
            feat_ext_op=feat_ext(x_in, getter=ema_getter, training=False),
            )


def main(argv):
    del argv  # Unused.
    # assert FLAGS.nu == 2
    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = DeepLP(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,

        beta=FLAGS.beta,
        w_match=FLAGS.w_match,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.epochs, FLAGS.imgs_per_epoch // FLAGS.batch, summary_interval=100)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
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

    FLAGS.set_default('num_pseudo_label_channels', 5)
    app.run(main)
