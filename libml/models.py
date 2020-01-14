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

import contextlib
# from contextlib import ContextManager




class BaseMultiModel(ClassifySemi):

    def __init__(self, *wargs, **kwargs):
        self.weights_dict = OrderedDict()
        super(BaseMultiModel, self).__init__(*wargs, **kwargs)
        

    @contextlib.contextmanager
    def collect_weights(self, layer_name):
        def get_current_variables():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        if layer_name not in self.weights_dict:
            self.weights_dict[layer_name] = []

        variable_list = get_current_variables()
        yield
        new_variable_list = get_current_variables()
        for var in new_variable_list:
            if var not in variable_list:
                self.weights_dict[layer_name].append(var)

    def get_kernel_by_layer(self, layer_name):
        assert layer_name in self.weights_dict
        v_list = [v for v in self.weights_dict[layer_name] if 'kernel' in v.name]
        assert len(v_list) == 1
        return v_list[0]


    def get_bias_by_layer(self, layer_name):
        assert layer_name in self.weights_dict
        v_list = [v for v in self.weights_dict[layer_name] if 'bias' in v.name]
        assert len(v_list) == 1
        return v_list[0]


    def print_weight_dict(self):
        for key, var_list in self.weights_dict.items():
            print(key)
            for var in var_list:
                print('\t', var.name, var.get_shape())

    @property
    def model_vars(self):
        all_var_list = []
        for key, var_list in self.weights_dict.items():
            all_var_list += var_list
        return all_var_list

    @property
    def model_var_names(self):
        return [v.name for v in self.model_vars]

    

class CNN13(BaseMultiModel):
    """Simplified reproduction of the Mean Teacher paper network. filters=128 in original implementation.
    Removed dropout, Gaussians, forked dense layers, basically all non-standard things."""

    def conv_out_layer(self, scales, filters, repeat, **kwargs):
        return 'bn3_3'

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



class ConvNet(BaseMultiModel):

    def conv_out_layer(self, scales, filters, repeat, **kwargs):
        return 'avgpool%d'%(scales+2)

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



class ResNet(BaseMultiModel):


    def conv_out_layer(self, scales, filters, repeat, **kwargs):
        return 'act'

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




class ResNet18(BaseMultiModel):

    def conv_out_layer(self, scales, filters, repeat, **kwargs):
        return 'act'

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
            ('maxpool1', partial(tf.layers.max_pooling2d, pool_size=3, strides=2)),
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

        layer_func_dict['fc1'] = partial(tf.layers.dense, units=self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        layer_func_dict['logits'] = tf.identity

        return layer_func_dict
 


class ShakeNet(BaseMultiModel):
    def conv_out_layer(self, scales, filters, repeat, **kwargs):
        return 'layer%d.%d'%(scales, repeat)

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
            layer_name = 'layer%d.%d' % (scale + 1, i + 1)
            if i == 1:
                layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale, stride=2 if scale else 1)
            else:
                layer_func_dict[layer_name] = partial(residual, name=layer_name, filters=filters<<scale)
        
        layer_func_dict['avgpool'] = partial(tf.reduce_mean, axis=[1, 2])
        layer_func_dict['fc1'] = partial(tf.layers.dense, units=self.nclass, kernel_initializer=tf.glorot_normal_initializer())

        return layer_func_dict



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

    def get_conv_out_layer(self, arch, **kwargs):
        return {
            self.MODEL_CNN13 : CNN13.conv_out_layer,
            self.MODEL_CONVNET : ConvNet.conv_out_layer,
            self.MODEL_RESNET : ResNet.conv_out_layer,
            self.MODEL_RESNET18 : ResNet18.conv_out_layer,
            self.MODEL_SHAKE : ShakeNet.conv_out_layer,
        }[arch](self, **kwargs)


    def classifier(self, x, arch, getter=None, verbose=False, **kwargs):
        ''' 对x进行卷积，并返回最后一层的输出结果 '''
        layer_func_dict = self.get_model(x, arch, getter=getter, **kwargs)
        out = x

        with tf.variable_scope(arch[0].upper()+arch[1:].lower(), reuse=tf.AUTO_REUSE, custom_getter=getter):
            for layer_name, layer_func in layer_func_dict.items():
                with self.collect_weights(layer_name):
                    out = layer_func(out)
                if verbose:
                    print('\t%s : '%layer_name, out.get_shape())
            return out


    def feature_ext(self, x, arch, feat_endpoint=None, getter=None, **kwargs):
        ''' 对x进行卷积，并返回指定层的输出结果 '''

        layer_func_dict = self.get_model(x, arch, getter=getter, **kwargs)

        default_feat_endpoint = 'fc1'
        if arch == self.MODEL_RESNET18:
            default_feat_endpoint = 'fc2'

        out = x
        with tf.variable_scope(arch[0].upper()+arch[1:].lower(), reuse=tf.AUTO_REUSE, custom_getter=getter):
            for layer_name, layer_func in layer_func_dict.items():
                if layer_name == default_feat_endpoint:
                    return out  
                
                with self.collect_weights(layer_name):
                    out = layer_func(out)
            
                if layer_name == feat_endpoint:
                    return out


    def features_ext(self, x, arch, getter=None, **kwargs):
        ''' 对x进行卷积，并返回每一层的输出结果 '''
        layer_func_dict = self.get_model(x, arch, getter=getter, **kwargs)

        out = x
        endpoint_dict = EasyDict()

        with tf.variable_scope(arch[0].upper()+arch[1:].lower(), reuse=tf.AUTO_REUSE, custom_getter=getter):
            for layer_name, layer_func in layer_func_dict.items():
                with self.collect_weights(layer_name):
                    out = layer_func(out)
                endpoint_dict[layer_name] = out
        return endpoint_dict


    def cam_ext(self, x, arch, getter=None, **kwargs):
        '''
            获取x的class activation map
        '''
        input_h = int(x.get_shape()[1])
        input_w = int(x.get_shape()[2])

        endpoints = self.features_ext(x, arch, getter=getter, **kwargs)
        cnn_output = endpoints[self.get_conv_out_layer(arch, **kwargs)]

        fc_weight = self.get_kernel_by_layer('fc1')
        cam_map = tf.einsum('bijf,fc->bijc', cnn_output, fc_weight)
        cam_map = tf.image.resize_images(cam_map, (input_w, input_h))

        return endpoints, cam_map


class BilinearModel(MultiModel):


    def get_fc_model(self, training, **kwargs):
        return OrderedDict([
                ('dropout', partial(tf.layers.dropout, rate = 0.8, training=training)),
                ('fc1', partial(tf.layers.dense, units=self.nclass, kernel_initializer=tf.glorot_normal_initializer())),
                ('logits', tf.identity),
        ])

    def apply(self, x, layer_func_dict, prefix=None, verbose=False, stop_at=''):
        prefix = prefix or ''
        out = x
        for layer_name, layer_func in layer_func_dict.items():
            with self.collect_weights(prefix+layer_name):
                out = layer_func(out)
            if verbose:
                print('\t%s : '%layer_name, out.get_shape())
            if  layer_name == stop_at:
                return out
        return out

    def classifier(self, x, arch, getter=None, verbose=False, **kwargs):
        ''' 对x进行卷积，并返回最后一层的输出结果 '''

        print(arch)
        if not arch.startswith('bcnn'):
            return super(BilinearModel, self).classifier(x, arch, getter, verbose=verbose, **kwargs)
        else:
            with tf.variable_scope(arch[0].upper()+arch[1:].lower(), reuse=tf.AUTO_REUSE, custom_getter=getter):
                branch_arch = arch.split('_')[1]
                layer_func_dict = self.get_model(x, branch_arch, getter=getter, **kwargs)
                cnn_out_layer = self.get_conv_out_layer(branch_arch, **kwargs)

                if verbose:
                    print('branch1:')
                with tf.variable_scope('branch1', reuse=tf.AUTO_REUSE, custom_getter=getter):
                    out1 = out2 = self.apply(x, layer_func_dict, verbose=verbose, stop_at=cnn_out_layer)

                if len(arch.split('_')[1:]) == 2:

                    branch_arch2 = arch.split('_')[2]
                    layer_func_dict2 = self.get_model(x, branch_arch2, getter=getter, **kwargs)
                    cnn_out_layer2 = self.get_conv_out_layer(branch_arch2, **kwargs)
                    
                    if verbose:
                        print('branch2:')
                    with tf.variable_scope('branch2', reuse=tf.AUTO_REUSE, custom_getter=getter):
                        out2 = self.apply(x, layer_func_dict2, verbose=verbose, stop_at=cnn_out_layer2)

                pooling_out = tf.einsum('ijkm,ijkn->imn', out1, out2)
                pooling_out = tf.layers.flatten(pooling_out)
                if verbose:
                    print('\t%s : '%'pooling', pooling_out.get_shape())
                
                fc_layer_dict = self.get_fc_model(**kwargs)


                return self.apply(pooling_out, fc_layer_dict, verbose=verbose,)




flags.DEFINE_enum('arch', MultiModel.MODEL_RESNET, MultiModel.MODELS + tuple(['bcnn_%s'%m for m in MultiModel.MODELS]) + tuple(['bcnn_%s_%s'%(m,m) for m in MultiModel.MODELS]), 'Architecture.')


