# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('hw_diff_final')
add_path('fpga_nips')

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'DNACoS'


"""Data Dir and Weight Dir"""
C.dataset_path = "/data1/ILSVRC/Data/CLS-LOC"

C.dataset = 'imagenet'

if C.dataset == 'cifar10':
    C.num_classes = 10
elif C.dataset == 'cifar100':
    C.num_classes = 100
elif C.dataset == 'imagenet':
    C.num_classes = 100
else:
    print('Wrong dataset.')
    sys.exit()


"""Image Config"""

C.num_train_imgs = 128000
C.num_eval_imgs = 50000

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""

C.opt = 'Sgd'

C.momentum = 0.9
C.weight_decay = 5e-4

C.betas=(0.5, 0.999)
C.num_workers = 32

""" Search Config """
C.grad_clip = 5

C.pretrain = False
# C.pretrain = 'ckpt/pretrain'

# C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
# C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
# C.stride_list = [1, 1, 2, 2, 1, 2, 1]

C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
C.stride_list = [1, 2, 2, 2, 1, 2, 1]

C.stem_channel = 16
C.header_channel = 1984

C.early_stop_by_skip = False

C.perturb_alpha = False
C.epsilon_alpha = 0.3

C.pretrain_epoch = 10

C.sample_func = 'gumbel_softmax'
C.temp_init = 5
C.temp_decay = 0.956

C.num_sample = 10

C.update_hw_freq = 5

C.hw_aware_nas = False
########################################


C.batch_size = 192
C.niters_per_epoch = int(C.num_train_imgs // C.batch_size * 0.8)
C.image_height = 224
C.image_width = 224
C.save = "search"

C.nepochs = 90
C.eval_epoch = 5

C.lr_schedule = 'cosine'
C.lr = 0.1
# linear 
C.decay_epoch = 20
# exponential
C.lr_decay = 0.97
# multistep
C.milestones = [50, 100, 200]
C.gamma = 0.1
# cosine
C.learning_rate_min = 0


########################################

C.train_portion = 0.8

C.unrolled = False

C.arch_learning_rate = 1e-2

C.efficiency_metric = 'flops'

C.flops_weight = 1e-9
C.flops_max = 4e8
C.flops_min = 2e8

C.edp_weight = 1e-2
C.edp_max = 2000
C.edp_min = 50