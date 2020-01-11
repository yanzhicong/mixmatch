
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

    print(DATASETS.keys())
    dataset = DATASETS['miniimagenet.2@40-50']()

    # img = dataset.labeled_data.get_img(0)
    # print(img.shape)

    # viewer = DatasourceViewer(dataset.labeled_data)

    img = dataset.labeled_data.get_img(0)

    img = np.array([img,]*10, dtype=np.uint8)

    viewer = ImageVSTensorData('test', img, img.astype(np.float32))
    
    draw_data_list_to_html([viewer], './test_dir')


if __name__ == "__main__":
    app.run(main)


