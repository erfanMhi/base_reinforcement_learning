from abc import ABC, abstractmethod
from ctypes import Union
from typing import Tuple

import gym
from gym.spaces import Space

from core.utils.util import retrieve_subclass
from core.utils.typing import ObsType, ActionType, ScalarType

class BaseEnv(ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod 
    def reset(self) -> Tuple[ObsType, dict]:
        pass

    @abstractmethod
    def _act(self) -> None:
        pass
    
    @abstractmethod
    def _observe(self) -> None:
        pass

    # enforcing this function to nnot change based
    def step(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, dict]:
        self._act(action)
        obs, reward, terminal, truncated, info = self._observe()
        return obs, reward, terminal, truncated, info

    @abstractmethod
    def observation_space(self) -> Space:
        pass

    @abstractmethod
    def action_space(self) -> Space:
        pass

    @classmethod
    def retrieve_env(cls, name: str) -> 'BaseEnv':
        return retrieve_subclass(cls, name)