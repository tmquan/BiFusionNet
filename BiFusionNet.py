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
class BiFusionNetModel(ModelDesc):
	def _get_inputs(self):
		pass

	def _collect_variables(self):
		pass

	def _get_optimizer(self):
		pass

	def _build_graph(self):
		pass

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
