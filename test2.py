
import functools
import os
import numpy as np
from absl import app
from absl import flags
from easydict import EasyDict
from libml import layers, utils, models
from libml.data import DATASETS, DATA_DIR
from libml.layers import MixMode
from libml.vis import *
from glob import glob

import tensorflow as tf
from tensorflow.python.client import device_lib


FLAGS = flags.FLAGS



def main(argv):
    dataset = DATASETS['miniimagenet.2@40-50']()

    viewer = DatasourceViewer(dataset.labeled_data)
    
    draw_data_list_to_html([viewer], os.path.join(DATA_DIR, 'miniimagenet_2_40'))


if __name__ == "__main__":
    app.run(main)