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
"""Classifier architectures."""

import functools
import itertools

import tensorflow as tf
from absl import flags

from libml import layers
from libml.train import ClassifySemi

from easydict import EasyDict
from collections import OrderedDict
from functools import partial



class CNN13(ClassifySemi):
    """Simplified reproduction of the Mean Teacher paper network. filters=128 in original implementation.
    Removed dropout, Gaussians, forked dense layers, basically all non-standard things."""

    # def __init__(*wargs, **kwargs):
    #     super(CNN13, self).__init__(*wargs, **kwargs)


    def classifier(self, x, scales, filters, training, getter=None, **kwargs):
        del kwargs
        assert scales == 3  # Only specified for 32x32 inputs.
        conv_args = dict(kernel_size=3, activation=tf.nn.leaky_relu, padding='same')
        bn_args = dict(training=training, momentum=0.999)

        endpoints = EasyDict()

        return OrderedDict([
            ('inp', lambda x:(x-self.dataset.mean)/self.dataset.std),
            ('conv1_1', partial(tf.layers.conv2d, filters=filters, **conv_args)),
            ('bn1_1', partial(tf.layers.batch_normalization, **bn_args)),
            ('conv1_2', partial(tf.layers.conv2d, filters=filters, **conv_args)),
            ('bn1_2', partial(tf.layers.batch_normalization, **bn_args)),
            ('pool1', partial(tf.layers.max_pooling2d, pool_size=2, strides=2)),

            ('conv2_1', partial(tf.layers.conv2d, filters=2*filters, **conv_args)),
            ('bn2_1', partial(tf.layers.batch_normalization, **bn_args)),
            ('conv2_2', partial(tf.layers.conv2d, filters=2*filters, **conv_args)),
            ('bn2_2', partial(tf.layers.batch_normalization, **bn_args)),
            ('conv2_3', partial(tf.layers.conv2d, filters=2*filters, **conv_args)),
            ('bn2_3', partial(tf.layers.batch_normalization, **bn_args)),
            ('pool2', partial(tf.layers.max_pooling2d, pool_size=2, strides=2)),

            ('conv3_1', partial(tf.layers.conv2d, filters=4*filters, **conv_args)),
            ('bn3_1', partial(tf.layers.batch_normalization, **bn_args)),
            ('conv3_2', partial(tf.layers.conv2d, filters=2*filters, **conv_args)),
            ('bn3_2', partial(tf.layers.batch_normalization, **bn_args)),
            ('conv3_3', partial(tf.layers.conv2d, filters=1*filters, **conv_args)),
            ('bn3_3', partial(tf.layers.batch_normalization, **bn_args)),
            
            ('pool3', partial(tf.reduce_mean, axis=[1, 2])),
            ('fc1', partial(tf.layers.dense, units=self.nclass)),
            ('logits', partial(tf.identity))
        ])



        # out = x

        # with tf.name_scope('CNN13'):
        #     with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):

        #         for layer_name, layer_func in layer_func_dict.items():
        #             out = layer_func(out)
        #             print('\t%s : '%layer_name, out.get_shape())
        #             endpoints[layer_name] = out

        #     return endpoints


class ConvNet(ClassifySemi):
    def classifier(self, x, scales, filters, getter=None, **kwargs):
        del kwargs
        conv_args = dict(kernel_size=3, activation=tf.nn.leaky_relu, padding='same')

        layer_func_dict = OrderedDict([
            ('inp', lambda x:(x-self.dataset.mean)/self.dataset.std),
            ('conv1', partial(tf.layers.conv2d, filters=filters, **conv_args)),
        ])

        for scale in range(scales):
            layer_func_dict['conv%d_1'%(scale+2)] = partial(tf.layers.conv2d, filters=filters<<scale, **conv_args)
            layer_func_dict['conv%d_2'%(scale+2)] = partial(tf.layers.conv2d, filters=filters<<(scale+1), **conv_args)
            layer_func_dict['avgpool%d'%(scale+2)] = partial(tf.layers.average_pooling2d, pool_size=2, strides=2)

        layer_func_dict['conv%d'%(scales+2)] = partial(tf.layers.conv2d, filters=self.nclass, kernel_size=3, padding='same')
        layer_func_dict['avgpool%d'%(scales+2)] = partial(tf.reduce_mean, axis=[1,2])
        layer_func_dict['logits'] = tf.identity

        return layer_func_dict

        
        # with tf.name_scope('ConvNet'):
        #     with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
        #         for layer_name, layer_func in layer_func_dict.items():
        #             out = layer_func(out)
        #             print('\t%s : '%layer_name, out.get_shape())
        #             endpoints[layer_name] = out
        #     return endpoints

        # with tf.name_scope('ConvNet'):
        #     with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
        #         y = tf.layers.conv2d(x, filters, **conv_args)
        #         for scale in range(scales):
        #             y = tf.layers.conv2d(y, filters << scale, **conv_args)
        #             y = tf.layers.conv2d(y, filters << (scale + 1), **conv_args)
        #             y = tf.layers.average_pooling2d(y, 2, 2)
        #         y = tf.layers.conv2d(y, self.nclass, 3, padding='same')
        #         logits = tf.reduce_mean(y, [1, 2])
        #     return logits


class ResNet(ClassifySemi):

    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        
        del kwargs
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, name, filters, stride=1, activate_before_residual=False):
            with tf.variable_scope(name):
                x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
                if activate_before_residual:
                    x0 = x

                x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
                x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
                x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

                if x0.get_shape()[3] != filters:
                    x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))

                return x0 + x


        layer_func_dict = OrderedDict([
            ('inp', lambda x:(x-self.dataset.mean)/self.dataset.std),
            ('conv1', partial(tf.layers.conv2d, filters=16, kernel_size=3, **conv_args(3,16))),
        ])

        for scale in range(scales):
            layer_name = 'res%da'%(scale+1)
            layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale, stride=(2 if scale else 1), activate_before_residual=(scale == 0))
            for i in range(repeat-1):
                layer_name = 'res%d%s'%(scale+1, chr(ord('b')+i))
                layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale)

        layer_func_dict['bn'] = partial(tf.layers.batch_normalization, **bn_args)
        layer_func_dict['act'] = leaky_relu
        layer_func_dict['avgpool'] = partial(tf.reduce_mean, axis=[1, 2])

        layer_func_dict['fc1'] = partial(tf.layers.dense, units=self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        layer_func_dict['logits'] = tf.identity

        return layer_func_dict


        # with tf.name_scope('ResNetSmall'):
        #     with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
        #         y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
                
        #         for scale in range(scales):
        #             y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
        #             for i in range(repeat - 1):
        #                 y = residual(y, filters << scale)

        #         y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
        #         y = tf.reduce_mean(y, [1, 2])
        #         logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())

        #     print('ResNet classifier, output : ', logits.get_shape())
        #     return logits


class ResNet18(ClassifySemi):

    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        lrelu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))


        def residual(x0, name, filters, stride=1, activate_before_residual=False):
            with tf.variable_scope(name):
                x = lrelu(tf.layers.batch_normalization(x0, **bn_args))
                if activate_before_residual:
                    x0 = x

                x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
                x = lrelu(tf.layers.batch_normalization(x, **bn_args))
                x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

                if x0.get_shape()[3] != filters:
                    x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))

                return x0 + x

        layer_func_dict = OrderedDict([
            ('inp', lambda x:(x-self.dataset.mean)/self.dataset.std),
            ('conv1', partial(tf.layers.conv2d, filters=64, kernel_size=7, strides=2, **conv_args(7,64))),
            ('maxpool1', partial(tf.layers.max_pooling2d, pool_size=2, strides=2)),
        ])

        for scale in range(scales):
            layer_name = 'res%da'%(scale+1)
            layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale, stride=(2 if scale else 1), activate_before_residual=(scale == 0))
            for i in range(repeat-1):
                layer_name = 'res%d%s'%(scale+1, chr(ord('b')+i))
                layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale)

        layer_func_dict['bn'] = partial(tf.layers.batch_normalization, **bn_args)
        layer_func_dict['act'] = lrelu
        layer_func_dict['avgpool'] = partial(tf.reduce_mean, axis=[1, 2])

        layer_func_dict['fc1'] = partial(tf.layers.dense, units=128, kernel_initializer=tf.glorot_normal_initializer())
        layer_func_dict['act'] = lrelu
        layer_func_dict['fc2'] = partial(tf.layers.dense, units=self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        layer_func_dict['logits'] = tf.identity

        return layer_func_dict
    

        # with tf.name_scope('ResNet'):
        #     with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
        #         for layer_name, layer_func in layer_func_dict.items():
        #             out = layer_func(out)
        #             print('\t%s : '%layer_name, out.get_shape())
        #             endpoints[layer_name] = out
        #     return endpoints

        # with tf.name_scope('ResNet'):
        #     with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):

        #         y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 64, 7, strides=2, **conv_args(7, 64))
        #         y = tf.layers.max_pooling2d(y, 2, 2)
        #         print('\tmax_pooling2d, output : ', y.get_shape())

        #         for scale in range(scales):
        #             y = residual(y, filters << scale, stride=(2 if scale else 1), activate_before_residual=(scale == 0))
        #             for i in range(repeat - 1):
        #                 y = residual(y, filters << scale)
                        
        #             # if 'record_feature' in kwargs and kwargs['record_feature']:
        #             #     self.features.append(y)
        #         y = lrelu(tf.layers.batch_normalization(y, **bn_args))
        #         print('\treduce_mean, input : ', y.get_shape())
        #         y = tf.reduce_mean(y, [1, 2])
                
        #         # if 'record_feature' in kwargs and kwargs['record_feature']:
        #         #     self.features.append(y)
        #         logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        #     print('ResNet classifier, output : ', logits.get_shape())
        #     # if 'record_feature' in kwargs and kgwars['record_feature']:
        #     #     self.logits = logits
        #     del kwargs
        #     return logits




class ShakeNet(ClassifySemi):
    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        del kwargs
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same', use_bias=False,
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, name, filters, stride=1):
            with tf.variable_scope(name):
                def branch():
                    x = tf.nn.relu(x0)
                    x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
                    x = tf.nn.relu(tf.layers.batch_normalization(x, **bn_args))
                    x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))
                    x = tf.layers.batch_normalization(x, **bn_args)
                    return x

                x = layers.shakeshake(branch(), branch(), training)

                if stride == 2:
                    x1 = tf.layers.conv2d(tf.nn.relu(x0[:, ::2, ::2]), filters >> 1, 1, **conv_args(1, filters >> 1))
                    x2 = tf.layers.conv2d(tf.nn.relu(x0[:, 1::2, 1::2]), filters >> 1, 1, **conv_args(1, filters >> 1))
                    x0 = tf.concat([x1, x2], axis=3)
                    x0 = tf.layers.batch_normalization(x0, **bn_args)
                elif x0.get_shape()[3] != filters:
                    x0 = tf.layers.conv2d(x0, filters, 1, **conv_args(1, filters))
                    x0 = tf.layers.batch_normalization(x0, **bn_args)

                return x0 + x

        layer_func_dict = OrderedDict([
            ('inp', lambda x:(x-self.dataset.mean)/self.dataset.std),
            ('conv1', partial(tf.layers.conv2d, filters=16, kernel_size=3, **conv_args(3, 16))),
        ])

        for scale, i in itertools.product(range(scales), range(repeat)):
            layer_name = 'layer%d.%d' % (scale + 1, i)
            if i == 0:
                layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale, stride=2 if scale else 1)
            else:
                layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale)
        
        layer_func_dict['avgpool'] = partial(tf.reduce_mean, axis=[1, 2])
        layer_func_dict['fc1'] = partial(tf.layers.dense, units=self.nclass, kernel_initializer=tf.glorot_normal_initializer())

        return layer_func_dict

        # with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
        #     y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
        #     for scale, i in itertools.product(range(scales), range(repeat)):
        #         with tf.variable_scope('layer%d.%d' % (scale + 1, i)):
        #             if i == 0:
        #                 y = residual(y, filters << scale, stride=2 if scale else 1)
        #             else:
        #                 y = residual(y, filters << scale)
        #     y = tf.reduce_mean(y, [1, 2])
        #     logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        # return logits


class MultiModel(CNN13, ConvNet, ResNet, ShakeNet, ResNet18):
    MODEL_CNN13, MODEL_CONVNET, MODEL_RESNET, MODEL_RESNET18, MODEL_SHAKE = 'cnn13 convnet resnet resnet18 shake'.split()
    MODELS = MODEL_CNN13, MODEL_CONVNET, MODEL_RESNET, MODEL_RESNET18, MODEL_SHAKE

    def augment(self, x, l, smoothing, **kwargs):
        del kwargs
        return x, l - smoothing * (l - 1. / self.nclass)

    def get_model(self, x, arch, getter=None, **kwargs):
        if arch == self.MODEL_CNN13:
            return CNN13.classifier(self, x, getter=getter, **kwargs)
        elif arch == self.MODEL_CONVNET:
            return ConvNet.classifier(self, x, getter=getter, **kwargs)
        elif arch == self.MODEL_RESNET:
            return ResNet.classifier(self, x, getter=getter, **kwargs)
        elif arch == self.MODEL_RESNET18:
            return ResNet18.classifier(self, x, getter=getter, **kwargs)
        elif arch == self.MODEL_SHAKE:
            return ShakeNet.classifier(self, x, getter=getter, **kwargs)
        else:
            raise ValueError('Model %s does not exists, available ones are %s' % (arch, self.MODELS))


    def classifier(self, x, arch, getter=None, **kwargs):

        layer_func_dict = self.get_model(x, arch, getter=getter, **kwargs)
        out = x

        with tf.name_scope(arch[0].upper()+arch[1:].lower()):
            with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
                for layer_name, layer_func in layer_func_dict.items():
                    out = layer_func(out)
                    print('\t%s : '%layer_name, out.get_shape())
            return out


    def feature_ext(self, x, arch, feat_endpoint=None, getter=None, **kwargs):
        layer_func_dict = self.get_model(x, arch, getter=getter, **kwargs)

        default_feat_endpoint = 'fc1'
        if arch == self.MODEL_RESNET18:
            default_feat_endpoint = 'fc2'

        out = x
        with tf.name_scope(arch[0].upper()+arch[1:].lower()):
            with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
                for layer_name, layer_func in layer_func_dict.items():
                    if layer_name == default_feat_endpoint:
                        # print('feature_ext : ', out.get_shape())
                        return out

                    out = layer_func(out)
                
                    if layer_name == feat_endpoint:
                        # print('feature_ext : ', out.get_shape())
                        return out


    def features_ext(self, x, arch, feat_endpoints, getter=None, **kwargs):
        layer_func_dict = self.get_model(x, arch, getter=getter, **kwargs)

        out = x
        endpoint_dict = EasyDict()

        with tf.name_scope(arch[0].upper()+arch[1:].lower()):
            with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
                for layer_name, layer_func in layer_func_dict.items():

                    out = layer_func(out)
                
                    if layer_name in feat_endpoints:
                        endpoint_dict[layer_name] = out
        return endpoint_dict


flags.DEFINE_enum('arch', MultiModel.MODEL_RESNET, MultiModel.MODELS, 'Architecture.')

