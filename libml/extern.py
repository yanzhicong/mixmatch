import os
import sys

from absl import flags

from libml import data, utils
from libml.models import MultiModel

FLAGS = flags.FLAGS

flags.DEFINE_integer('steps_per_epoch', 60000//64, 'Steps per epoch.')


class ClassifySemiWithPLabel(MultiModel):
    
    def train_step(self, train_session, data_labeled, data_unlabeled, summary=None):
            x, y = self.session.run([data_labeled, data_unlabeled])
            if summary is not None:
                _, s, self.tmp.step = train_session.run([self.ops.train_op, summary, self.ops.update_step],
                                                feed_dict={self.ops.x: x['image'],
                                                            self.ops.y: y['image'],
                                                            self.ops.label: x['label'],
                                                            self.ops.pseudo_label:y['pseudo_label']})
                self.summary_writer.add_summary(s, global_step=self.tmp.step)
            else:
                self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
                                                feed_dict={self.ops.x: x['image'],
                                                            self.ops.y: y['image'],
                                                            self.ops.label: x['label'],
                                                            self.ops.pseudo_label:y['pseudo_label']})[1]

    def update_pseudo_label(self):
        raise NotImplementedError()

    def eval_pseudo_label(self, dataset):
        raise NotImplementedError()

    def add_summaries(self, feed_extra=None, **kwargs):
        super(ClassifySemiWithPLabel, self).add_summaries(feed_extra=feed_extra, **kwargs)

        def gen_stats():
            return self.eval_pseudo_label(dataset=self.dataset)

    def on_epoch_start(self, epoch_ind, epochs):
        super(ClassifySemiWithPLabel, self).on_epoch_start(epoch_ind, epochs)

        update = False
        if epoch_ind < 10:
            update = True
        if epoch_ind < 

        self.update_pseudo_label()

