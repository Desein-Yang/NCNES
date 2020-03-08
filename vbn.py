#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   vbn.py
@Time    :   2020/02/01 21:19:51
@Version :   1.0
@Describtion:   Virtual Batch Normalization
"""

# here put the import lib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from preprocess import ProcessUnit

import numpy as np


class VirtualBatchNorm2D(nn.Module):
    """Virtual Batch Normalization layer.  
    """

    def __init__(self, num_features):
        super(VirtualBatchNorm2D, self).__init__()
        self.named_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.bn = nn.BatchNorm2d(num_features, momentum=1.0, affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight)
        init.zeros_(self.bias)
        self.mean.zero_()
        self.var.fill_(1)

    def set_mean_var(self, mean, var):
        self.mean = mean
        self.var = var

    def set_mean_var_from_bn(self, bn):
        # setup reference batch's mean and var
        self.mean = bn.running_mean
        self.var = bn.running_var

    def forward(self, input):
        self._check_input_dim(input)
        batch_size = input.size()[0]
        new_coeff = 1.0 / (batch_size + 1)
        old_coeff = 1.0 - new_coeff
        output = self.bn(input)
        new_mean = self.bn.running_mean
        new_var = self.bn.running_var
        mean = new_coeff * new_mean + old_coeff * self.mean
        var = new_coeff * new_var + old_coeff * self.var
        return F.batch_norm(input, mean, var, self.weight, self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class VirtualBatchNorm1D(nn.Module):
    def __init__(self, num_features):
        super(VirtualBatchNorm1D, self).__init__()
        self.named_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight)
        init.zeros_(self.bias)
        self.mean.zero_()
        self.var.fill_(1)

    def set_mean_var(self, mean, var):
        self.mean = mean
        self.var = var

    def set_mean_var_from_bn(self, bn):
        # setup reference batch's mean and var
        self.mean = bn.running_mean
        self.var = bn.running_var

    def forward(self, input):
        # using reference batch's mean and var
        self._check_input_dim(input)
        return F.batch_norm(input, self.mean, self.var, self.weight, self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError("expected 2D input (got {}D input)".format(input.dim()))


def one_explore_for_vbn(env, prob, args):
    """collect 1 reference with random actions of VBN  
    Args:  
        prob:   Select probability = 0.01  
        env:    Game environment for evaluation  
    Returns:\n
        r:      One reference of vbn
    """
    r = []
    env.frameskip = 1
    observation = env.reset()
    break_is_true = False
    ProcessU = ProcessUnit()
    ProcessU.step(observation)
    ep_max_step = 108000
    # no_op_frames = np.random.randint(1, 31)
    no_op_frames = np.random.randint(1, 6) * 6
    n_action = env.action_space.n

    for i in range(no_op_frames):
        observation, reward, done, _ = env.step(np.random.randint(n_action))
        ProcessU.step(observation)
        if np.random.rand() <= prob:
            r.append(ProcessU.to_torch_tensor())

    for step in range(ep_max_step):
        action = np.random.randint(n_action)
        for i in range(args.FRAME_SKIP):
            observation, reward, done, _ = env.step(action)
            ProcessU.step(observation)
            if np.random.rand() <= prob:
                r.append(ProcessU.to_torch_tensor())
            if done:
                break_is_true = True
        if break_is_true or len(r) > args.refer_batch_size:
            break
    return r


def explore_for_vbn(env, prob, args):
    """Collect all reference fo VBN  
    Args:  
        prob:   Select probability. Default 0.01.  
        env:    Game environment for evaluation  
        args:   Reference batch size = 128  
    Returns:  
        reference:  Visual batch normalization reference  
    """
    max_time = 1000
    return_r = []
    for i in range(max_time):
        one_time_r = one_explore_for_vbn(env, prob, args)
        return_r.extend(one_time_r)
        if len(return_r) > args.refer_batch_size:
            break
    return return_r[: args.refer_batch_size]
