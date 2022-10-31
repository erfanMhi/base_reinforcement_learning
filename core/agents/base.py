from abc import ABC, abstractmethod

from torch import optim
from torch import nn

from core.utils.exploration.epsilon_greedy import EpsilonGreedyExploration
from core.nets.fc import FullyConnectedNet
from core.utils.typing import ActionType, ObsType
from core.utils.util import retrieve_subclass

optimizers = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
}

loss = {
    'mse': nn.MSELoss,
}

# TODO: all the create functions should turn into registry or 
class BaseAgent(ABC):

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._cfg = cfg
        self._device = self._cfg.device
        self._action_space = self._cfg.action_space
        self._obs_space = self._cfg.obs_space

    @abstractmethod
    def act(self, obs: ObsType) -> ActionType:
        pass

    @abstractmethod
    def learn(self, obs: ObsType, action: ActionType, reward: float, next_obs: ObsType, terminal: bool) -> None:
        pass
    
    def _create_optimizer(self, params: nn.Module, name: str, **args) -> optim.Optimizer: #TODO: write registry for optimizers
        optimizer = optimizers[name]
        return optimizer(params, **args)

    def _create_loss(self, name: str, **args) -> nn.Module: #TODO: write registry for losses
        loss_func = loss[name]
        return loss_func(**args)

    def _create_exploration(self, name, **args) -> None: #TODO: write registry for exploration approaches
        if name == 'epsilon-greedy':
            return EpsilonGreedyExploration(**args)
        else:
            raise ValueError(f"Exploration type {name} not implemented")

    def _create_network(self, name: str, **args) -> nn.Module: #TODO: write registry for networks
        
        if name == 'fully-connected':
            return FullyConnectedNet.create_by_gym_space(self._obs_space, self._action_space, **args)
        else:
            raise NotImplementedError(f"Network type {name} not implemented")

    @classmethod
    def retrieve_agent(cls, name: str) -> 'BaseAgent':
        return retrieve_subclass(cls, name)

    @abstractmethod
    def turn_eval_mode_on(self) -> None: ...
    
    @abstractmethod
    def turn_eval_mode_off(self) -> None: ...