import os
import sys

from absl import flags

from libml import data, utils
from libml.models import MultiModel

FLAGS = flags.FLAGS

import numpy as np

# flags.DEFINE_integer('steps_per_epoch', 60000//64, 'Steps per epoch.')


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

    def eval_pesudo_label_acc(self, plabel, weight, true_label):
        """Computes the precision and weighted precision of pesudo-label"""

        acc = float(np.sum((plabel == true_label).astype(np.int32))) / float(len(plabel))
        weighted_acc = float(np.sum((plabel == true_label).astype(np.float32) * weight)) / float(np.sum(weight))

        print("Pesudo Label Acc: %0.5f"%(acc * 100.0))
        print("Pesudo Label Weighted Acc: %0.5f"%(weighted_acc * 100.0))

        weight = np.sort(weight)[::-1]
        wi = [1, 1000, 2000, 4000, 8000, 16000, 32000]
        ind_str_list = [str(ind) for ind in wi]
        weight_str_list = ["%0.5f"%weight[ind] for ind in wi]

        print("Pesudo Label Weight(" + "/".join(ind_str_list) + "): " + "/".join(weight_str_list))

        return acc, weighted_acc



    def on_epoch_start(self, epoch_ind, epochs):
        super(ClassifySemiWithPLabel, self).on_epoch_start(epoch_ind, epochs)

        update = False
        # if epoch_ind < 10:
        #     update = True
        # if epoch_ind < 

        self.update_pseudo_label()

