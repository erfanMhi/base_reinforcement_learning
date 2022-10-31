from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn
from gym.spaces import Space
from gym.spaces.utils import flatdim

from core.utils.configs import BaseConfig

activation_map = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'softmax': nn.Softmax,
    None: lambda: lambda x: x
}

class BaseNet(nn.Module, ABC):
        
    @abstractmethod
    def forward(self, state, action):
        pass

    @staticmethod
    def get_activation(activation_name: str) -> nn.Module:
        return activation_map[activation_name]()

    @classmethod
    @abstractmethod
    def create_by_gym_space(cls, obs_space: Space , action_space: Space, **kwargs) -> 'BaseNet':
        pass