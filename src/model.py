#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/02/01 17:04:47
@Author  :   Qi Yang
@Version :   1.0
@Describtion:   Build network for atari games. 

'''

# here put the import lib
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vbn import VirtualBatchNorm2D, VirtualBatchNorm1D
from collections import deque


class ESNet(nn.Module):
    """
    Network module for Atari Games which is a combination of DQN and CES.

    Ref:Chrabaszcz, Patryk, Ilya Loshchilov, and Frank Hutter. "Back to basics: Benchmarking canonical evolution strategies for playing atari." arXiv preprint arXiv:1802.08842 (2018).
    Url:https://arxiv.org/abs/1802.08842
    """
    def __init__(self, ARGS):
        super(ESNet, self).__init__()
        self.conv1_f = 32
        self.conv2_f = 64
        self.conv3_f = 64
        
        self.conv1 = nn.Conv2d(ARGS.FRAME_SKIP, self.conv1_f, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(self.conv1_f, self.conv2_f, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(self.conv2_f, self.conv3_f, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.conv1_f, affine=False)
        self.bn2 = nn.BatchNorm2d(self.conv2_f, affine=False)
        self.bn3 = nn.BatchNorm2d(self.conv3_f, affine=False)
        self.bn4 = nn.BatchNorm1d(512, affine=False)

        self.vbn1 = VirtualBatchNorm2D(self.conv1_f)
        self.vbn2 = VirtualBatchNorm2D(self.conv2_f)
        self.vbn3 = VirtualBatchNorm2D(self.conv3_f)
        self.vbn4 = VirtualBatchNorm1D(512)

        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, ARGS.action_n)

        self.set_parameter_no_grad()
        self._initialize_weights()
        self.status = "bn"
        # self.previous_frame should be PILImage
        self.previous_frame = None

    def forward_bn(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)

        x = x.view(-1, 7*7*64) 
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def forward_vbn(self, x):
        x = self.vbn1(self.conv1(x))
        x = F.relu(x)
        x = self.vbn2(self.conv2(x))
        x = F.relu(x)
        x = self.vbn3(self.conv3(x))
        x = F.relu(x)
        x = x.view(-1, 7*7*64)
        x = self.vbn4(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def switch_to_vbn(self):
        self.vbn1.set_mean_var_from_bn(self.bn1)
        self.vbn2.set_mean_var_from_bn(self.bn2)
        self.vbn3.set_mean_var_from_bn(self.bn3)
        self.vbn4.set_mean_var_from_bn(self.bn4)
        self.status = 'vbn'

    def switch_to_bn(self):
        self.status = 'bn'
    
    def forward(self, x):
        if self.status == 'bn':
            return self.forward_bn(x)
        elif self.status == 'vbn':
            return self.forward_vbn(x)

    def _initialize_weights(self):
        for m in self.modules():
            # Orthogonal initialization and layer scaling
            # Paper name : Implementation Matters in Deep Policy Gradient: A case study on PPO and TRPO
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                #nn.init.orthogonal_(m.weight)
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_parameter_no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_size(self):
        """
        Returns:   
            Number of all params
        """
        count = 0
        for params in self.parameters():
            count += params.numel()
        return count

    def get_name_slice_dict(self):
        """Get a dict whose keys are all params name and values are params index   
        Returns:   
            {
                'conv1.weight':[left,right],   
                'conv2.weight':[left,right]
            }   
            where left(right) is start(end) index of params conv1.weight in 1-d array
        """
        d = dict()
        for name, param in self.named_parameters():
            d[name] = param.numel()
        slice_dict = {}
        value_list = list(d.values())
        names_list = list(d.keys())
        left, right = 0, 0
        for ind, name in enumerate(names_list):
            right += value_list[ind]
            slice_dict[name] = [left, right]
            left += value_list[ind]
        return slice_dict

def build_model(ARGS):
    """
    Args:
        action_n(int):    Available action numbers of one game
    Returns:   
        ESNet(nn.Module): Neural network module with specified architecture
    """
    if ARGS.env_type == "atari":
        return ESNet(ARGS)

def build_sigma(model: torch.nn.Module, ARGS):
    """Build a dict to store sigma of all params.  
    Args:  
        model(nn.Module):   Network module of offspring.
        ARGS:               Sigma init value.
    Returns:  
        sigma_dict(dict):   Dict of sigma of all params.
    Init:  
        ones_like tensor * sigma_init.
    """
    sigma_dict = {}
    for name, parameter in model.named_parameters():
        sigma_dict[name] = torch.ones_like(parameter,dtype = torch.float) * ARGS.sigma_init
    return sigma_dict

def build_mean(model: torch.nn.Module,ARGS):
    """Build a dict to store mean of all params.  
    Args:  
        model(nn.Module):   Network module of offspring.
        ARGS:               High limit and low limit
    Returns:  
        mean_dict(dict):    Dict of mean of all params.  
    Init:
        mean= L + (H-L) *rand
    """
    mean_dict = {}
    for name, parameter in model.named_parameters():
        mean_dict[name] = torch.ones_like(parameter,dtype=torch.float) * ARGS.L + (ARGS.H - ARGS.L) * torch.rand_like(parameter,dtype=torch.float)
        mean_dict[name] = torch.clamp(mean_dict[name],ARGS.L,ARGS.H)
    return mean_dict
