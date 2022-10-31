from ctypes import Union
from typing import Tuple

import gym
from gym.spaces import Space

from core.envs import BaseEnv
from core.utils.configs.base import BaseConfig
from core.utils.typing import ActionType, ObsType

class CartPoleEnv(BaseEnv):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._env = gym.make("CartPole-v1")
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._last_action = None

    def reset(self) -> Tuple[ObsType, dict]:
        return self._env.reset()

    def _act(self, action: ActionType) -> None:
        self._last_action = action

    def _observe(self) -> Tuple[ObsType, float, bool, bool, dict]:
        return self._env.step(self._last_action)

    @property
    def observation_space(self) -> Space:
        return self._env.observation_space
    
    @property
    def action_space(self) -> Space:
        return self._env.action_space