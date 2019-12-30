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
"""Training loop, checkpoint saving and loading, evaluation code."""

import json
import os.path
import shutil

import numpy as np
import tensorflow as tf
from absl import flags
from easydict import EasyDict
from tqdm import trange
import threading
from collections import OrderedDict

from libml import data, utils

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_float('lr_decay_rate', 0.001, 'Learning rate decay rate.')
flags.DEFINE_integer('decay_start_epoch', 20, 'Decay start epoch.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('epochs', 100, 'Training epochs.')
flags.DEFINE_integer('steps_per_epoch', 60000//64, 'Steps per epoch.')
flags.DEFINE_integer('imgs_per_epoch', 60000, 'images per epoch.')
flags.DEFINE_integer('save_kimg', 64, 'Save checkpoint period in kibi-samples.')
flags.DEFINE_integer('keep_ckpt', 50, 'Number of checkpoints to keep.')
flags.DEFINE_string(
    'eval_ckpt', '', 'Checkpoint to evaluate. If provided, do not do training, just do eval.')


class Model:


    def __init__(self, train_dir: str, dataset: data.DataSet, **kwargs):
        self.train_dir = os.path.join(
            train_dir, self.experiment_name(**kwargs))

        self.params = EasyDict(kwargs)
        self.dataset = dataset
        self.session = None
        self.tmp = EasyDict(print_queue=[], cache=EasyDict())
        self.step = tf.train.get_or_create_global_step()
        self.epoch = tf.get_variable('global_epoch', shape=[], dtype=tf.int32, initializer=tf.zeros_initializer(), trainable=False)

        self.learning_rate = utils.get_exponential_learning_rate(lr=FLAGS.lr, global_epoch = self.epoch, start_epoch=FLAGS.decay_start_epoch, total_epochs=FLAGS.epochs, decay_rate=FLAGS.lr_decay_rate)
        kwargs['lr'] = self.learning_rate

        self.ops = self.model(**kwargs)

        self.ops.update_step = tf.assign_add(self.step, 1)
        self.add_summaries(**kwargs)

        print(' Config '.center(80, '-'))
        print('train_dir', self.train_dir)
        print('%-32s %s' % ('Model', self.__class__.__name__))
        print('%-32s %s' % ('Dataset', dataset.name))
        for k, v in sorted(kwargs.items()):
            print('%-32s %s' % (k, v))
        print(' Model '.center(80, '-'))
        to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)])
                          for v in utils.model_vars(None)]
        to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
        sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
        fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
        for x in to_print[:-1]:
            print(fmt % x)
        print()
        print(fmt % to_print[-1])
        print('-' * 80)
        self._create_initial_files()

    @property
    def arg_dir(self):
        return os.path.join(self.train_dir, 'args')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    def train_print(self, text):
        self.tmp.print_queue.append(text)

    def _create_initial_files(self):
        for dir in (self.checkpoint_dir, self.arg_dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.save_args()

    def _reset_files(self):
        shutil.rmtree(self.train_dir)
        self._create_initial_files()

    def save_args(self, **extra_params):
        with open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
            json.dump({**self.params, **extra_params},
                      f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, train_dir):
        with open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
            params = json.load(f)
        instance = cls(train_dir=train_dir, **params)
        instance.train_dir = train_dir
        return instance

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    def eval_mode(self, ckpt=None):
        self.session = tf.Session(config=utils.get_config())
        saver = tf.train.Saver()
        if ckpt is None:
            ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
        else:
            ckpt = os.path.abspath(ckpt)
        saver.restore(self.session, ckpt)
        self.tmp.step = self.session.run(self.step)
        print('Eval model %s at global_step %d' %
              (self.__class__.__name__, self.tmp.step))
        return self

    def model(self, **kwargs):
        raise NotImplementedError()

    def add_summaries(self, **kwargs):
        raise NotImplementedError()


class ClassifySemi(Model):
    """Semi-supervised classification."""

    def __init__(self, train_dir: str, dataset: data.DataSet, nclass: int, **kwargs):
        self.nclass = nclass
        self.train_step_monitor_summary = None

        Model.__init__(self, train_dir, dataset, nclass=nclass, **kwargs)

    def train_step(self, train_session, data_labeled, data_unlabeled=None, summary=None):
        if data_unlabeled is not None:
            x, y = self.session.run([data_labeled, data_unlabeled])
            if summary is not None:
                _, s, self.tmp.step = train_session.run([self.ops.train_op, summary, self.ops.update_step],
                                                feed_dict={self.ops.x: x['image'],
                                                            self.ops.y: y['image'],
                                                            self.ops.label: x['label']})
                self.summary_writer.add_summary(s, global_step=self.tmp.step)
            else:
                self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
                                                feed_dict={self.ops.x: x['image'],
                                                            self.ops.y: y['image'],
                                                            self.ops.label: x['label']})[1]
        else:
            x = self.session.run([data_labeled,])[0]
            if summary is not None:
                _, s, self.tmp.step = train_session.run([self.ops.train_op, summary, self.ops.update_step],
                                                feed_dict={self.ops.x: x['image'],
                                                            self.ops.label: x['label']})
                self.summary_writer.add_summary(s, global_step=self.tmp.step)
            else:
                self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
                                                feed_dict={self.ops.x: x['image'],
                                                            self.ops.label: x['label']})[1]

    def on_epoch_start(self, epoch_ind, epochs):
        pass

    def on_epoch_end(self, epoch_ind, epochs):
        pass

    def on_training_interrupted(self, epoch_ind, epochs):
        pass



    def train(self, epochs, steps_per_epoch, summary_interval=100, ssl=True):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        def get_dataset_reader(dataset, batch, prefetch=16):
            d = dataset.batch(batch).prefetch(
                prefetch).make_initializable_iterator()
            return d.initializer, d.get_next()

        with tf.name_scope('dataset_reader'):
            self.train_labeled = get_dataset_reader(self.dataset.train_labeled, batch)
            if ssl:
                self.train_unlabeled = get_dataset_reader(self.dataset.train_unlabeled, batch)
            else:
                self.train_unlabeled = None, None
            self.eval_labeled = get_dataset_reader(self.dataset.eval_labeled, batch)
            self.valid = get_dataset_reader(self.dataset.valid, batch)
            self.test = get_dataset_reader(self.dataset.test, batch)

        self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir, graph=tf.get_default_graph())

        with tf.Session(config=utils.get_config()) as train_session:
            self.session = train_session
            self.session.run(tf.global_variables_initializer())
            self.tmp.step = self.session.run(self.step)

            # run dataset initializer
            if self.train_unlabeled[0] is not None:
                self.session.run([self.train_labeled[0], self.train_unlabeled[0]])
            else:
                self.session.run([self.train_labeled[0],])


            for epoch_ind in range(epochs):
                self.session.run(self.epoch.assign(epoch_ind))

                self.on_epoch_start(epoch_ind, epochs)

                loop = trange(0, steps_per_epoch*batch, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + epoch_ind, epochs))

                for step_ind in loop:
                    if step_ind % summary_interval == 0:
                        self.train_step(train_session, self.train_labeled[1], self.train_unlabeled[1], self.train_step_monitor_summary)
                    else:
                        self.train_step(train_session, self.train_labeled[1], self.train_unlabeled[1])


                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))

                s, st = self.session.run([self.summary, self.step])
                self.summary_writer.add_summary(s, global_step=st)

                self.on_epoch_end(epoch_ind, epochs)

            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))


    def tune(self, train_nimg):
        batch = FLAGS.batch

        self.session.run([self.train_labeled[0], self.train_unlabeled[0]])
        for _ in trange(0, train_nimg, batch, leave=False, unit='img', unit_scale=batch, desc='Tuning'):
            x, y = self.session.run([self.train_labeled[1], self.train_unlabeled[1]])
            self.session.run([self.ops.tune_op], feed_dict={self.ops.x: x['image'],
                                                            self.ops.y: y['image'],
                                                            self.ops.label: x['label']})

    def eval_checkpoint(self, ckpt=None):
        self.eval_mode(ckpt)
        # self.cache_eval()
        raw = self.eval_stats(classify_op=self.ops.classify_raw)
        ema = self.eval_stats(classify_op=self.ops.classify_op)
        self.tune(16384)
        tuned_raw = self.eval_stats(classify_op=self.ops.classify_raw)
        tuned_ema = self.eval_stats(classify_op=self.ops.classify_op)
        print('%16s %8s %8s %8s' % ('', 'labeled', 'valid', 'test'))
        print('%16s %8s %8s %8s' % (('raw',) + tuple('%.2f' % x for x in raw)))
        print('%16s %8s %8s %8s' % (('ema',) + tuple('%.2f' % x for x in ema)))
        print('%16s %8s %8s %8s' % (('tuned_raw',) + tuple('%.2f' % x for x in tuned_raw)))
        print('%16s %8s %8s %8s' % (('tuned_ema',) + tuple('%.2f' % x for x in tuned_ema)))

    
    def eval_stats(self, eval_dict=None, batch=None, feed_extra=None, classify_op=None):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []
        weighted_accuracies = []

        if eval_dict is None:
            eval_dict = OrderedDict(
                train_labeled=self.eval_labeled,
                valid=self.valid,
                test=self.test)

        for subset, dataset in eval_dict.items():
            predicted = []
            labels = []

            self.session.run([dataset[0]])
            while True:
                try:
                    v = self.session.run(dataset[1])
                    p = self.session.run(
                        classify_op,
                        feed_dict={
                            self.ops.x: v['image'],
                            **(feed_extra or {})
                        })
                    predicted.append(p)
                    labels.append(v['label'])
                    
                except tf.errors.OutOfRangeError:
                    break
            predicted = np.concatenate(predicted, axis=0)
            labels = np.concatenate(labels, axis=0)

            accuracies.append((predicted.argmax(1) == labels).mean() * 100)
            weighted_accuracies.append(utils.cal_entropy_weighed_acc(labels, predicted))

        self.train_print('kimg %-5d  accuracy train/valid/test  %.2f  %.2f  %.2f  weighted accuracy train/valid/test  %.2f  %.2f  %.2f' %
                        tuple([self.tmp.step >> 10] + accuracies + weighted_accuracies))
        return np.array(accuracies, 'f'), np.array(weighted_accuracies, 'f')


    def add_summaries(self, feed_extra=None, **kwargs):
        del kwargs

        def gen_stats():
            return self.eval_stats(feed_extra=feed_extra)

        accuracies = tf.py_func(gen_stats, [], [tf.float32, tf.float32])
        self.summary = tf.summary.merge([    
                tf.summary.scalar('accuracy/train_labeled', accuracies[0][0]),
                tf.summary.scalar('accuracy/valid', accuracies[0][1]),
                tf.summary.scalar('accuracy/test', accuracies[0][2]),
                tf.summary.scalar('weighted_accuracy/train_labeled', accuracies[1][0]),
                tf.summary.scalar('weighted_accuracy/valid', accuracies[1][1]),
                tf.summary.scalar('weighted_accuracy/test', accuracies[1][2]),
                ])

