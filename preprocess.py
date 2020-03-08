#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   preprocess.py
@Time    :   2020/02/01 16:55:58
@Describtion:  Preprocessing procedures to transfrom Atari raw screen pixel to input of network 
"""

# here put the import lib
import torch
import numpy as np

from torchvision import transforms
from collections import deque


trans = transforms.Compose(
    [transforms.Grayscale(), transforms.Resize((84, 84)), transforms.ToTensor()]
)


class ProcessUnit(object):
    """
    Preprocessing procedures to transfrom Atari raw screen pixel to input of network   
        1. Take themaximum value for each pixel colour   
        2. Grayscale and resize 84x84   
        3. Turn 4(or 3) frame into 1 tensor 84x84x4   
    Ref: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.\n
    Url: https://www.nature.com/articles/nature14236   
    
    Attributes:\n
        self.length(int):       Length of queue (length * frame_skip)
        self.frame_list(list):  Queue of frame
    """

    def __init__(self, length=4, frame_skip=4):
        self.length = length * frame_skip
        self.frame_list = deque(maxlen=self.length)

    def step(self, x):
        """add a frame to frame_list"""
        # insert in left
        self.frame_list.appendleft(x)

    def to_torch_tensor(self):
        """Turn 4 frames into 1 tensor (84x84x4)"""
        length = len(self.frame_list)
        x_list = []
        i = 0
        while i < length:
            if i == length - 1:
                x_list.append(self.transform(self.frame_list[i]))
            else:
                x = np.maximum(self.frame_list[i], self.frame_list[i + 1])
                x_list.append(self.transform(x))
            i += 4
        while len(x_list) < 4:
            x_list.append(x_list[-1])
        return torch.cat(x_list, 1)

    def transform(self, x):
        """pytorch transform"""
        x = transforms.ToPILImage()(x).convert("RGB")
        x = trans(x)
        x = x.reshape(1, 1, 84, 84)
        return x
