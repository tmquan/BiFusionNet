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

from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_3d, conv_3d_transpose
from tflearn.layers.core import dropout
from tflearn.layers.merge_ops import merge
from tflearn.activations import linear, sigmoid, tanh, elu 


from tensorpack import (FeedfreeTrainerBase, QueueInput, ModelDesc, DataFlow)
from tensorpack import *

from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbolic_functions

###############################################################################
# Global variables
BATCH_SIZE = 1
EPOCH_SIZE = 16
NB_FILTERS = 8

LAMBDA = 1e+2

DIMX = 256
DIMY = 256
DIMZ = 128
DIMC = 1


def generator_fusionnet(images, name='generator'):
	dimx = DIMX
	dimy = DIMY
	dimz = DIMZ

	with tf.variable_scope(name):
		# return images
		e1 = conv_3d(incoming=images, 
					 nb_filter=NB_FILTERS*1, 
					 filter_size=4,
					 strides=[1, 1, 1, 1, 1], # DIMZ/1, DIMY/2, DIMX/2, 
					 regularizer='L1',
					 activation='elu')
		e1 = batch_normalization(incoming=e1)
		###
		e2 = conv_3d(incoming=e1, 
					 nb_filter=NB_FILTERS*1, 
					 filter_size=4,
					 strides=[1, 2, 2, 2, 1], # DIMZ/2, DIMY/4, DIMX/4, 
					 regularizer='L1',
					 activation='elu')
		
		e2 = batch_normalization(incoming=e2)
		###
		e3 = conv_3d(incoming=e2, 
					 nb_filter=NB_FILTERS*2, 
					 filter_size=4,
					 strides=[1, 2, 2, 2, 1], # DIMZ/4, DIMY/8, DIMX/8,
					 regularizer='L1',
					 activation='elu')
		e3 = batch_normalization(incoming=e3)
		###
		e4 = conv_3d(incoming=e3, 
					 nb_filter=NB_FILTERS*2, 
					 filter_size=4,
					 strides=[1, 2, 2, 2, 1], # DIMZ/8, DIMY/16, DIMX/16,
					 regularizer='L1',
					 activation='elu')
		e4 = batch_normalization(incoming=e4)
		###
		e5 = conv_3d(incoming=e4, 
					 nb_filter=NB_FILTERS*4, 
					 filter_size=4,
					 strides=[1, 2, 2, 2, 1], # DIMZ/16, DIMY/32, DIMX/32,
					 regularizer='L1',
					 activation='elu')
		e5 = batch_normalization(incoming=e5)		
		###
		e6 = conv_3d(incoming=e5, 
					 nb_filter=NB_FILTERS*4, 
					 filter_size=4,
					 strides=[1, 2, 2, 2, 1], # DIMZ/32, DIMY/64, DIMX/64,
					 regularizer='L1',
					 activation='elu')
		e6 = batch_normalization(incoming=e6)		
		###
		e7 = conv_3d(incoming=e6, 
					 nb_filter=NB_FILTERS*8, 
					 filter_size=4,
					 strides=[1, 2, 2, 2, 1], # DIMZ/64, DIMY/128, DIMX/128,
					 regularizer='L1',
					 activation='elu')
		e7 = batch_normalization(incoming=e7)		
		### Middle
		e8 = conv_3d(incoming=e7, 
					 nb_filter=NB_FILTERS*8, 
					 filter_size=4,
					 strides=[1, 2, 2, 2, 1], # DIMZ/128, DIMY/256, DIMX/256,
					 regularizer='L1',
					 activation='elu')
		# print "Dim8: ", dimz, dimy, dimx
		dimz, dimy, dimx = dimz/2, dimy/2, dimx/2
		e8 = batch_normalization(incoming=e8)		

		################### Decoder

		# print "Dim D7a: ", dimz, dimy, dimx
		d7 = conv_3d_transpose(incoming=e8, 
							   nb_filter=NB_FILTERS*8, 
							   filter_size=4,
							   strides=[1, 2, 2, 2, 1], # DIMZ/64, DIMY/128, DIMX/128,
							   regularizer='L1',
							   activation='elu', 
							   output_shape=[2, 4, 4])

		d7 = batch_normalization(incoming=d7)
		
		d7 = dropout(incoming=d7, keep_prob=0.5)
		
		d7 = merge(tensors_list=[d7, e7], mode='elemwise_sum')
		# d7 = d7+e7	
		###
		d6 = conv_3d_transpose(incoming=d7, 
							   nb_filter=NB_FILTERS*4, 
							   filter_size=4,
							   strides=[1, 2, 2, 2, 1], # DIMZ/32, DIMY/64, DIMX/64,
							   regularizer='L1',
							   activation='elu', 
							   output_shape=[4, 8, 8])
		d6 = batch_normalization(incoming=d6)	
		d6 = dropout(incoming=d6, keep_prob=0.5)
		
		d6 = merge(tensors_list=[d6, e6], mode='elemwise_sum')
		# d6 = d6+e6
		###
		d5 = conv_3d_transpose(incoming=d6, 
							   nb_filter=NB_FILTERS*4, 
							   filter_size=4,
							   strides=[1, 2, 2, 2, 1], # DIMZ/16, DIMY/32, DIMX/32,
							   regularizer='L1',
							   activation='elu', 
							   output_shape=[8, 16, 16])
		d5 = batch_normalization(incoming=d5)	
		d5 = dropout(incoming=d5, keep_prob=0.5)
		
		d5 = merge(tensors_list=[d5, e5], mode='elemwise_sum')
		# d5 = d5+e5
		###
		d4 = conv_3d_transpose(incoming=d5, 
							   nb_filter=NB_FILTERS*2, 
							   filter_size=4,
							   strides=[1, 2, 2, 2, 1], # DIMZ/8, DIMY/16, DIMX/16,
							   regularizer='L1',
							   activation='elu', 
							   output_shape=[16, 32, 32])
		d4 = batch_normalization(incoming=d4)	
		
		d4 = merge(tensors_list=[d4, e4], mode='elemwise_sum')
		# d4 = d4+e4
		###
		d3 = conv_3d_transpose(incoming=d4, 
							   nb_filter=NB_FILTERS*2, 
							   filter_size=4,
							   strides=[1, 2, 2, 2, 1], # DIMZ/4, DIMY/8, DIMX/8,
							   regularizer='L1',
							   activation='elu', 
							   output_shape=[32, 64, 64])
		d3 = batch_normalization(incoming=d3)	
		
		d3 = merge(tensors_list=[d3, e3], mode='elemwise_sum')
		# d3 = d3+e3
		###
		d2 = conv_3d_transpose(incoming=d3, 
							   nb_filter=NB_FILTERS*1, 
							   filter_size=4,
							   strides=[1, 2, 2, 2, 1], # DIMZ/2, DIMY/4, DIMX/4,
							   regularizer='L1',
							   activation='elu', 
							   output_shape=[64, 128, 128])
		d2 = batch_normalization(incoming=d2)	
		
		d2 = merge(tensors_list=[d2, e2], mode='elemwise_sum')
		# d2 = d2+e2
		
		###
		d1 = conv_3d_transpose(incoming=d2, 
							   nb_filter=NB_FILTERS*1, 
							   filter_size=4,
							   strides=[1, 2, 2, 2, 1], # DIMZ/1, DIMY/2, DIMX/2,
							   regularizer='L1',
							   activation='elu', 
							   output_shape=[128, 256, 256])
		d1 = batch_normalization(incoming=d1)	
		
		d1 = merge(tensors_list=[d1, e1], mode='elemwise_sum')
		# d1 = d1+e1
		###
		
		out = conv_3d_transpose(incoming=d1, 
							   nb_filter=1, 
							   filter_size=4,
							   strides=[1, 1, 1, 1, 1], # DIMZ/1, DIMY/1, DIMX/1,
							   regularizer='L1',
							   activation='tanh', 
							   output_shape=[128, 256, 256])
		return out, e8


###############################################################################
class BiFusionNetModel(ModelDesc):
	def _get_inputs(self):
		return [InputDesc(tf.float32, [None, DIMZ, DIMY, DIMX, DIMC], 'image'),
				InputDesc(tf.float32, [None, DIMZ, DIMY, DIMX, DIMC], 'label')] # if 1 AtoB, if 0 BtoA

	def _collect_variables(self, gG_scope='gG', gF_scope='gF'):
		self.gG_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, gG_scope)
		self.gF_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, gF_scope)

	@auto_reuse_variable_scope
	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

	@auto_reuse_variable_scope
	def generator_G(self, imgs):
		with tf.device('/gpu:0'):
			return generator_fusionnet(imgs, name='gG')
		# pass

	@auto_reuse_variable_scope
	def generator_F(self, imgs):
		with tf.device('/gpu:1'):
			return generator_fusionnet(imgs, name='gF')

	def _build_graph(self, inputs):
		image, label = inputs
		image = (image / 255.0 - 0.5)*2.0 # From (0, 255) to (0, 1) to (-0.5, 0.5) to (-1, 1)
		label = (label / 255.0 - 0.5)*2.0 # From (0, 255) to (0, 1) to (-0.5, 0.5) to (-1, 1)
		
		X = image
		Y = label
		###############################################################################################
		with tf.name_scope('BiFusion'):
			Y_ , G1 = self.generator_G(X)
			X__, F1 = self.generator_F(Y_)

			X_ , F2 = self.generator_F(Y)
			Y__, G2 = self.generator_G(X_)
		with tf.name_scope("BiFusion_loss"):
			#http://devdocs.io/tensorflow~python/tf/reduce_mean
			self.Xo_loss   = tf.reduce_mean(tf.abs(X__ - X), name='Xo_loss')
			self.Xb_loss   = tf.reduce_mean(tf.abs(G1 - F1), name='Xb_loss')
			self.Yo_loss   = tf.reduce_mean(tf.abs(Y__ - Y), name='Yo_loss')
			self.Yb_loss   = tf.reduce_mean(tf.abs(G2 - F2), name='Yb_loss')
			self.L1_loss   = self.Xo_loss + self.Yo_loss + LAMBDA*(self.Xb_loss + self.Yb_loss)
		###############################################################################################
		# tensorboard visualization            
		viz_cycleG = tf.concat([image[:,0:1,:,:,0],Y_[:,0:1,:,:,0],X__[:,0:1,:,:,0]], 3, name='viz_cycleG')
		viz_cycleF = tf.concat([label[:,0:1,:,:,0],X_[:,0:1,:,:,0],Y__[:,0:1,:,:,0]], 3, name='viz_cycleF')
		viz_concat = tf.concat([viz_cycleG, viz_cycleF], 2, name='viz_concat')
		viz_concat = (viz_concat / 2.0 + 0.5) * 255.0
		viz_concat = tf.reshape(viz_concat, [1, 512, 768, 1])
		tf.summary.image('concatenation', viz_concat, max_outputs=30)

		viz_genG = (Y_ / 2.0 + 0.5) * 255.0
		viz_genG = tf.cast(tf.clip_by_value(viz_genG, 0, 255), tf.uint8, name='viz_genG') 
		# tf.summary.image('genG', viz_genG[:,0:1,:,:,0], max_outputs=30)

		viz_genF = (X_ / 2.0 + 0.5) * 255.0
		viz_genF = tf.cast(tf.clip_by_value(viz_genF, 0, 255), tf.uint8, name='viz_genF') 
		# tf.summary.image('genF', viz_genF[:,0:1,:,:,0], max_outputs=30)


		# Collect all the variable loss 
		add_moving_summary(self.Xo_loss, self.Xb_loss, 
						   self.Yo_loss, self.Yb_loss, 
						   self.L1_loss
						   )
		self._collect_variables()
		
###############################################################################
class BiFusionNetTrainer(FeedfreeTrainerBase):
	def __init__(self, config):
		self._input_method = QueueInput(config.dataflow)
		super(BiFusionNetTrainer, self).__init__(config)

	def _setup(self):
		super(BiFusionNetTrainer, self)._setup()
		self.build_train_tower()
		optG = self.model._get_optimizer()
		optF = self.model._get_optimizer()

		#
		self.gG_min = optG.minimize(self.model.L1_loss, var_list=self.model.gG_vars, name='gG_op')
		self.gF_min = optF.minimize(self.model.L1_loss, var_list=self.model.gF_vars, name='gF_op')
		
		self.train_op = [self.gG_min, self.gF_min]
		# self.train_op = opt.minimize(self.model.L1_loss, var_list=[self.model.gF_vars, self.model.gG_vars], name='g_op')
###############################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32'):
		"""
		Args:
			shapes (list): a list of lists/tuples. Shapes of each component.
			size (int): size of this DataFlow.
			random (bool): whether to randomly generate data every iteration.
				Note that merely generating the data could sometimes be time-consuming!
			dtype (str): data type.
		"""
		# super(FakeData, self).__init__()

		self.dtype  = dtype
		self.imageDir = imageDir
		self.labelDir = labelDir
		self._size  = size

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)   
		pass

	def random_flip(self, image, seed=None):
		assert image.ndim == 4
		if seed:
			np.random.seed(seed)
		random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[::1,::1,::-1,::1]
			image = flipped
		elif random_flip==2:
			flipped = image[::1,::-1,::1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[::1,::-1,::-1,::1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image
		
	def random_reverse(self, image, seed=None):
		assert image.ndim == 4
		if seed:
			np.random.seed(seed)
		random_reverse = np.random.randint(1,2)
		if random_reverse==1:
			flipped = image[::1,::1,::1]
		elif random_reverse==2:
			flipped = image[::1,::1,::-1]
		image = flipped
		return image

	def random_square_rotate(self, image, seed=None):
		assert image.ndim == 4
		if seed:
			np.random.seed(seed)		
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image

	def random_elastic(self, image, seed=None):
		assert image.ndim == 4
		import random
		if seed:
			random.seed(seed)
			np.random.seed(seed)
		shape = image.shape
		dimx, dimy = shape[1], shape[2]
		size = random.choice(range(2,4)) #8
		ampl = random.choice(range(2,4)) #8
		du = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		dv = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		# Done distort at boundary
		du[ 0,:] = 0
		du[-1,:] = 0
		du[:, 0] = 0
		du[:,-1] = 0
		dv[ 0,:] = 0
		dv[-1,:] = 0
		dv[:, 0] = 0
		dv[:,-1] = 0
		import cv2
		from scipy.ndimage.interpolation 	import map_coordinates
		# Interpolate du
		DU = cv2.resize(du, (shape[1], shape[2])) 
		DV = cv2.resize(dv, (shape[1], shape[2])) 
		X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]))
		indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
		
		warped = image.copy()
		for z in range(image.shape[-1]): #Loop over the channel
			# print z
			imageZ = np.squeeze(image[...,z])
			flowZ  = map_coordinates(imageZ, indices, order=1).astype(np.float32)

			warpedZ = flowZ.reshape(image[...,z].shape)
			warped[...,z] = warpedZ
		return warped

	def random_shuffle_label(self, label, k=15, seed=None):
		# Groundtruth index (for fast computing)
		# k = 15
		colorFactor = -(-256//k)
		colorRange  = k
		shuffles = np.arange(0, 15)
		linearly = np.arange(0, 15)
		# print shuffles
		# print linearly
		if seed:
			np.random.seed(seed)
		np.random.shuffle(shuffles)
		shuffled = np.zeros_like(label)
		for value in linearly:
		    # Map the sorted local to shuffle index which is k_colorable k = 15
		    shuffled[label==(colorFactor*value)] = colorFactor*shuffles[value]
		return shuffled

	def get_data(self, shuffle=True):
		# self.reset_state()
		images = glob.glob(self.imageDir + '/*.tif')
		labels = glob.glob(self.labelDir + '/*.tif')
		# if shuffle:
		# 	import random
		# 	images = random.shuffle(images)
		# 	labels = random.shuffle(labels)
		# print images
		# print labels
		# EPOCH_SIZE = len(images)
		# EPOCH_SIZE = 30
		for k in range(self._size):
			from random import randrange
			rand_index_image = randrange(0, len(images))
			rand_index_label = randrange(0, len(labels))
			image = skimage.io.imread(images[rand_index_image])
			label = skimage.io.imread(labels[rand_index_label])
			
			# Change z, x, y to x, y, z
			image = np.transpose(image, (1, 2, 0))
			label = np.transpose(label, (1, 2, 0))


			image = np.expand_dims(image, axis=0)
			label = np.expand_dims(label, axis=0)

			seed = np.random.randint(0, 2015)

			# #TODO: augmentation here
			# Image Augmentation
			# image = self.random_flip(image, seed=seed)
			# image = self.random_reverse(image, seed=seed)
			# image = self.random_square_rotate(image, seed=seed)
			image = self.random_elastic(image, seed=seed).astype(np.uint8)
			# image = skimage.util.random_noise(image, seed=seed)
			image = skimage.img_as_ubyte(image)
			
			# Label Augmentation
			# label = self.random_flip(label, seed=seed)
			# label = self.random_reverse(label, seed=seed)
			# label = self.random_square_rotate(label, seed=seed)
			label = self.random_shuffle_label(label, k=15, seed=seed)
			label = self.random_elastic(label, seed=seed)
			label = skimage.img_as_ubyte(label)

			# print image.shape
			# print label.shape
			# Change n, x, y, z to  n, z, x, y
			image = np.transpose(image, (0, 3, 1, 2))
			label = np.transpose(label, (0, 3, 1, 2))



			image = np.expand_dims(image, axis=-1)
			label = np.expand_dims(label, axis=-1)

			# print image.shape
			# print label.shape
			yield [image.astype(np.uint8), label.astype(np.uint8)]

###############################################################################
def get_data():
	ds_train = ImageDataFlow(args.imageDir, args.labelDir, EPOCH_SIZE)
	ds_valid = ImageDataFlow(args.imageDir, args.labelDir, EPOCH_SIZE)
	PrintData(ds_train, num=1)
	PrintData(ds_valid, num=1)
	# ds_train = PrefetchDataZMQ(ds_train, nr_proc=4)
	# ds_valid = PrefetchDataZMQ(ds_valid, nr_proc=4)
	ds_train = PrefetchData(ds_train, 16)
	ds_valid = PrefetchData(ds_valid, 16)
	return ds_train, ds_valid

def get_config():
	logger.auto_set_dir()
	# dataset = get_data()
	ds_train, ds_valid = get_data()
	ds_train.reset_state()
	ds_valid.reset_state() 
	#print ds_train.size()
	return TrainConfig(
		dataflow=ds_train,
		callbacks=[
			PeriodicTrigger(ModelSaver(), every_k_epochs=100),
			ScheduledHyperParamSetter('learning_rate', [(100, 1e-4), (200, 1e-2), (300, 1e-5), (400, 2e-6), (500, 1e-6)]),
			],
		model=BiFusionNetModel(),
		steps_per_epoch=ds_train.size(),
		max_epoch=4000,
	)
def sample(imageDir, labelDir, model_path):
	import skimage.io
	import time
	pred_config = PredictConfig(
		session_init=get_model_loader(model_path),
		model=CycleGANModel(),
		input_names=['image', 'label'],
		output_names=['viz_concat', 'viz_genG', 'viz_genF'])
		# output_names=['genF'])
	imageFiles = glob.glob(imageDir + '/*.png')
	labelFiles = glob.glob(labelDir + '/*.png')
	images = []
	labels = []
	pairs  = []
	for imageFile, labelFile in zip(imageFiles, labelFiles):
		image = skimage.io.imread(imageFile)
		label = skimage.io.imread(labelFile)
		image = np.expand_dims(image, axis=-1)
		label = np.expand_dims(label, axis=-1)
		image = np.expand_dims(image, axis=0)
		label = np.expand_dims(label, axis=0)
		pairs.append([image, label])
	ds   = DataFromList(lst=pairs, shuffle=False)
	pred = SimpleDatasetPredictor(pred_config, ds)
	

	# result = []
	generatedG = []
	generatedF = []
	for o in pred.get_result():
		genG = o[1] #[:, :, :, ::-1]
		genF = o[2] #[:, :, :, ::-1]
	# 	print genG.shape
	# 	generatedG.append(o[1])
		generatedG.append(genG)
		generatedF.append(genF)

	generatedG = np.array(generatedG)
	generatedF = np.array(generatedF)
	# print generatedG.dtype
	# print generatedG.shape

	prefix = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
	skimage.io.imsave(prefix+'_generatedG.tif', generatedG.astype(np.uint8))
	skimage.io.imsave(prefix+'_generatedF.tif', generatedF.astype(np.uint8))
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
