#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: BiFusionNet.py
# Author: tmquan
###############################################################################
import os
import cv2
import numpy as np 
import skimage.io  
import glob
import pickle
import sys
import shutil
import argparse

import tensorflow as tf


from tensorpack import (FeedfreeTrainerBase, QueueInput, ModelDesc, DataFlow)
from tensorpack import *

from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbolic_functions

###############################################################################
# Global variables
BATCH_SIZE = 1
EPOCH_SIZE = 64
NB_FILTERS = 8

DIMX = 256
DIMY = 256
DIMZ = 128
DIMC = 1

###############################################################################

def tf_batch_norm(input_var, epsilon=1e-5, decay=0.9, mode='3d', name="batch_norm_2d", reuse=False):
    axes = [0, 1, 2]
    if mode == '3d':
        axes = [0, 1, 2, 3]
    elif mode == '2d':
        axes = [0, 1, 2]
    elif mode == '1d':
        axes = [0, 1]
 
    with tf.variable_scope(name, reuse=reuse):
        shape = input_var.get_shape().as_list()
        beta = tf.get_variable("beta", [shape[-1]],
                               initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        batch_mean, batch_var = tf.nn.moments(input_var, axes, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        normed = tf.nn.batch_normalization(input_var, mean, var, beta, gamma, epsilon, name="batch_normalization")
        return normed

def tf_dropout(input_var, drop_prob=0.5, name="dropout"):
    return tf.nn.dropout(input_var, 1-drop_prob, name=name)

def tf_conv3d(input_, output_dim, 
			  k_d=5, k_h=5, k_w=5, 
			  strides=[1, 1, 1, 1, 1],
			  stddev=0.02, dropout=True,
              name="conv3d", reuse=False, activation=None, padding='SAME', bn=True):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=strides, padding=padding)
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if dropout:
        	conv = tf_dropout(conv)
        if bn:
            conv = tf_batch_norm(conv, mode='3d')
        if activation is not None:
            conv = activation(conv)
        return conv

def tf_deconv3d(input_, output_shape,
                k_d=5, k_h=5, k_w=5, 
                strides=[1, 1, 1, 1, 1],
                stddev=0.02, bn=True, dropout=False,
                name="deconv3d", padding='VALID', reuse=False, activation=None):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        batch_size = tf_get_batch_size(input_)
        # output_shape = to_list(output_shape)
        w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=[batch_size, ] + output_shape,
                                        strides=strides, padding=padding)
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv = tf.reshape(deconv, [batch_size, ] + output_shape)
        if bn:
            deconv = tf_batch_norm(deconv, mode='3d')
        if activation is not None:
            deconv = activation(deconv)
        if dropout:
        	deconv = tf_dropout(deconv)
        return deconv


def tf_bottleneck(input_var, output_dim, 
				  name="bottleneck", reuse=False, activation=None):
	with tf.variable_scope(name, reuse=reuse):
		indentity = tf.identity(input_var, name="identity")
		conv_1x1i = tf_conv3d(input_var=input_var, output_dim=output_dim, 
							  k_d=1, k_h=1, k_w=1, 
							  activation=tf.nn.elu, name="conv_1x1i")
		conv_3x3m = tf_conv3d(input_var=conv_1x1i, output_dim=output_dim, 
							  k_d=1, k_h=1, k_w=1, 
							  activation=tf.nn.elu, name="conv_3x3m")
		conv_1x1o = tf_conv3d(input_var=conv_3x3m, output_dim=output_dim, 
							  k_d=1, k_h=1, k_w=1, 
							  activation=tf.nn.elu, name="conv_1x1o")
		summation = tf.add(identity, conv_1x1o, name="summation")
		activated = activation(summation, name="activated")
		return activated

def get_fusion(images, name="fusion"):
	with tf.variable_scope(name):
		enc1a = tf_conv3d(images, 		name="enc1a", activation=tf.nn.elu, output_dim=NB_FILTERS*1, strides=[1, 1, 2, 2, 1])
		res1a = tf_bottleneck(enc1a, 	name="res1a", activation=tf.nn.elu, output_dim=NB_FILTERS*1)
		
		enc2a = tf_conv3d(res1a, 		name="enc2a", activation=tf.nn.elu, output_dim=NB_FILTERS*2, strides=[1, 2, 2, 2, 1])
		res2a = tf_bottleneck(enc2a, 	name="res2a", activation=tf.nn.elu, output_dim=NB_FILTERS*2)
		
		enc3a = tf_conv3d(res2a, 		name="enc3a", activation=tf.nn.elu, output_dim=NB_FILTERS*4, strides=[1, 2, 2, 2, 1])
		res3a = tf_bottleneck(enc3a, 	name="res3a", activation=tf.nn.elu, output_dim=NB_FILTERS*4)
		
		enc4a = tf_conv3d(res3a, 		name="enc4a", activation=tf.nn.elu, output_dim=NB_FILTERS*8, strides=[1, 2, 2, 2, 1])
		res4a = tf_bottleneck(enc4a, 	name="res4a", activation=tf.nn.elu, output_dim=NB_FILTERS*8)

		enc5a = tf_conv3d(res4a, 		name="enc5a", activation=tf.nn.elu, output_dim=NB_FILTERS*8, strides=[1, 2, 2, 2, 1])
		res5a = tf_bottleneck(enc5a, 	name="res5a", activation=tf.nn.elu, output_dim=NB_FILTERS*8)

		enc6a = tf_conv3d(res5a, 		name="enc6a", activation=tf.nn.elu, output_dim=NB_FILTERS*8, strides=[1, 2, 2, 2, 1])
		res6a = tf_bottleneck(enc6a, 	name="res6a", activation=tf.nn.elu, output_dim=NB_FILTERS*8)

		enc7a = tf_conv3d(res6a, 		name="enc7a", activation=tf.nn.elu, output_dim=NB_FILTERS*8, strides=[1, 2, 2, 2, 1])
		res7a = tf_bottleneck(enc7a, 	name="res7a", activation=tf.nn.elu, output_dim=NB_FILTERS*8)

		bridge = tf.identity(res7a, 	name="bridge")

		res7b = tf_bottleneck(bridge, 	name="res7b", activation=tf.nn.elu, output_dim=NB_FILTERS*8)		
		dec7b = tf_deconv3d(res7b, 		name="dec7b", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, NB_FILTERS*8])
		dec7b = tf.add(dec7b, enc7a, 	name="long7")

		res6b = tf_bottleneck(dec7b, 	name="res6b", activation=tf.nn.elu, output_dim=NB_FILTERS*8)		
		dec6b = tf_deconv3d(res6b, 		name="dec6b", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, NB_FILTERS*8])
		dec6b = tf.add(dec6b, enc6a, 	name="long6")

		res5b = tf_bottleneck(dec6b, 	name="res5b", activation=tf.nn.elu, output_dim=NB_FILTERS*8)		
		dec5b = tf_deconv3d(res5b, 		name="dec5b", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, NB_FILTERS*8])
		dec5b = tf.add(dec5b, enc5a, 	name="long5")

		res4b = tf_bottleneck(dec5b, 	name="res4b", activation=tf.nn.elu, output_dim=NB_FILTERS*8)		
		dec4b = tf_deconv3d(res4b, 		name="dec4b", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, NB_FILTERS*8])
		dec4b = tf.add(dec4b, enc4a, 	name="long4")

		res3b = tf_bottleneck(dec4b, 	name="res3b", activation=tf.nn.elu, output_dim=NB_FILTERS*4)		
		dec3b = tf_deconv3d(res3b, 		name="dec3b", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, NB_FILTERS*4]e)
		dec3b = tf.add(dec3b, enc3a, 	name="long3")

		res2b = tf_bottleneck(dec3b, 	name="res2b", activation=tf.nn.elu, output_dim=NB_FILTERS*2)		
		dec2b = tf_deconv3d(res2b, 		name="dec2b", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, NB_FILTERS*2]e)
		dec2b = tf.add(dec2b, enc2a, 	name="long2")

		res1b = tf_bottleneck(dec2b, 	name="res1b", activation=tf.nn.elu, output_dim=NB_FILTERS*1)		
		dec1b = tf_deconv3d(res1b, 		name="dec1b", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, NB_FILTERS*1]e)
		dec1b = tf.add(dec1b, enc1a, 	name="long1")

		labels = tf_deconv3d(res1b, 	name="labels", activation=tf.nn.elu, output_shape=[BATCH_SIZE, DIMZ, DIMY, DIMX, DIMC])

		return labels, bridge
###############################################################################
class BiFusionNetModel(ModelDesc):
	def _get_inputs(self):
		pass

	def _collect_variables(self):
		pass

	def _get_optimizer(self):
		pass

	def _build_graph(self):
		image, label = inputs
		image = (image / 255.0 - 0.5)*2.0
		label = (label / 255.0 - 0.5)*2.0
		
		X = image
		Y = label

###############################################################################
class BiFusionNetTrainer(FeedfreeTrainerBase):
	def __init__(self, config):
		pass

	def _setup(self):
		pass

###############################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32'):
		pass

	def size(self):
		pass

###############################################################################

###############################################################################
if __name__ == '__main__':
	#https://docs.python.org/3/library/argparse.html
	parser = argparse.ArgumentParser()
	#
	parser.add_argument('--gpu', 		help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 		help='load models for continue train or predict')
	parser.add_argument('--sample',		help='run sampling one instance', action='store_true')
	parser.add_argument('--imageDir',   help='Image directory', required=True)
	parser.add_argument('--labelDir',   help='Label directory', required=True)
	global args
	args = parser.parse_args() # Create an object of parser

	#
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if args.sample:
		sample(args.imageDir, args.labelDir, args.load)
	else:
		config = get_config()
		if args.load:
			config.session_init = SaverRestore(args.load)
		BiFusionNetTrainer(config).train()
