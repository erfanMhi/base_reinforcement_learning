from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from gym.spaces import Space
from gym.spaces.utils import flatdim

from core.nets.base import BaseNet


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self._initialize(self.linear)
        self.activation = BaseNet.get_activation(activation)

    @staticmethod
    def _initialize(layer: nn.Linear) -> None:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.activation(self.linear(x))

class FullyConnectedNet(BaseNet):

    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int], activation: str = 'relu', output_activation: str = None) -> None:
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_layers = hidden_layers
        self._activation = activation
        self._output_activation = output_activation
        
        fc_layers = self._create_network_modules()

        self._model = nn.Sequential(*fc_layers)

    def _create_network_modules(self) -> List[nn.Module]:
        prev_layer_size = self._input_size
        fc_layers = []
        for layer_size in self._hidden_layers:
            layer = FullyConnectedLayer(prev_layer_size, layer_size, self._activation)
            fc_layers.append(layer)
            prev_layer_size = layer_size
        output_layer = FullyConnectedLayer(prev_layer_size, self._output_size, self._output_activation)
        fc_layers.append(output_layer)
        return fc_layers
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._model(obs)

    def get_layers_weights_and_bias(self) -> Dict[str, torch.Tensor]:
        layers_weights_and_bias = {}
        layer_count = 0
        for layer in self._model:
            if isinstance(layer, FullyConnectedLayer):
                tag = f'layer-{layer_count}'
                layers_weights_and_bias[f'weights-{tag}'] = layer.linear.weight
                layers_weights_and_bias[f'bias-{tag}'] = layer.linear.bias
        return layers_weights_and_bias

    @classmethod
    def create_by_gym_space(cls, obs_space: Space , action_space: Space, **kwargs) -> 'FullyConnectedNet':
        input_size = flatdim(obs_space)
        output_size = flatdim(action_space)
        return cls(input_size, output_size, **kwargs)