import os
import sys


sys.path.append('./')

import tarfile
import tempfile
from urllib import request
# import wget
import subprocess
import numpy as np
import cv2
from tqdm import tqdm
import json

from easydict import EasyDict
from libml.data import DATA_DIR



train_folders = {
    'mini-imagenet':[
        'E:\\Data\\mini-imagenet\\train',
        'E:\\Data\\mini-imagenet\\test',
        '/mnt/data02/dataset/mini-imagenet-cls/train',
        '/mnt/data02/dataset/mini-imagenet-cls/test',
    ],
    'tiered-imagenet':[

    ]
}


def load_folder(folder):
    image_list = []
    label_list = []
    for class_name in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, class_name)):
            for file in os.listdir(os.path.join(folder, class_name)):
                if file.split('.')[-1] in ['jpg', 'jpeg']:
                    image_list.append(os.path.join(folder, class_name, file))
                    label_list.append(class_name)
    return image_list, label_list




def _load_miniimagenet():

    image_filepath_list = []
    image_label_list = []
    for folder in train_folders['mini-imagenet']:
        if os.path.exists(folder):
            fl, ll = load_folder(folder)
            image_filepath_list += fl
            image_label_list += ll


    train_image_list = []
    train_label_list = []
    test_image_list = []
    test_label_list = []

    nb_test_images_per_class = 100

    class_name_list = np.unique(image_label_list)
    class_ind_dict = {class_name:ind for ind, class_name in enumerate(class_name_list)}
    ind_class_dict = {ind:class_name for class_name, ind in class_ind_dict.items()}

    for class_name in class_name_list:
        class_filepath_list = [fp for fp, l in zip(image_filepath_list, image_label_list) if l == class_name]

        test_list = list(np.random.choice(class_filepath_list, size=nb_test_images_per_class, replace=False))
        train_list = [fp for fp in class_filepath_list if fp not in test_list]

        train_image_list += train_list
        train_label_list += [class_ind_dict[class_name] for _ in train_list]
        test_image_list += test_list
        test_label_list += [class_ind_dict[class_name] for _ in test_list]

    print('Total nb train images : ', len(train_image_list))
    print('Total nb test images : ', len(test_image_list))

    mean_list = []
    var_list = []
    for image_fp in tqdm(train_image_list):
        img = cv2.imread(image_fp).astype(np.float32) / 255.0
        if img is not None:
            mean_list.append(np.mean(img, axis=(0, 1)))
            var_list.append(np.var(img, axis=(0, 1)))

    img_mean = np.mean(np.array(mean_list), axis=0)
    img_var = np.mean(np.array(var_list), axis=0)

    return {
        'train' : {
            'images' : train_image_list,
            'labels' : train_label_list,
        },
        'test' : {
            'images' : test_image_list,
            'labels' : test_label_list,
        },
        'meanvar' : {
            'mean' : img_mean.tolist(),
            'var' : img_var.tolist(),
        },
        'clsmap' : ind_class_dict
    }


def _save_as_txt(data, filename):
    with open(os.path.join(DATA_DIR, filename+'.txt'), 'w') as outfile:
        for image_filepath, label in zip(data['images'], data['labels']):
            outfile.write('%s,%s\n'%(image_filepath, str(label)))



CONFIGS = dict(
    miniimagenet=dict(loader=_load_miniimagenet),
    # cifar100=dict(loader=_load_cifar100,
    #               checksums=dict(train=None, test=None)),
    # svhn=dict(loader=_load_svhn,
    #           checksums=dict(train=None, test=None, extra=None)),
    # stl10=dict(loader=_load_stl10,
    #            checksums=dict(train=None, test=None)),
)

if __name__ == '__main__':
    if len(sys.argv[1:]):
        subset = set(sys.argv[1:])
    else:
        subset = set(CONFIGS.keys())
    try:
        os.makedirs(DATA_DIR)
    except OSError:
        pass
    for name, config in CONFIGS.items():
        if name not in subset:
            continue

        # if 'is_installed' in config:
        #     if config['is_installed']():
        #         print('Skipping already installed:', name)
        #         continue
        # elif _is_installed(name, config['checksums']):
        #     print('Skipping already installed:', name)
        #     continue


        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_txt)

        for sub_name, data in datas.items():
            print(sub_name)
            if sub_name == 'readme':
                filename = os.path.join(DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with open(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(DATA_DIR, file_and_data.filename)
                    open(path, "wb").write(file_and_data.data)
            elif sub_name == 'meanvar' or sub_name == 'clsmap':
                path = os.path.join(DATA_DIR, '%s-%s.json' % (name, sub_name))
                with open(path, 'w') as f:
                    f.write(json.dumps(data, indent=4))
            else:
                saver(data, '%s-%s' % (name, sub_name))
