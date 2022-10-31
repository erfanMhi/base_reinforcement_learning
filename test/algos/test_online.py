import logging
from pathlib import Path
from random import random
import shutil
from types import SimpleNamespace

import pytest
import numpy as np
from core.agents.base import BaseAgent
from gym import spaces

from core.algos import BaseAlgo
from core.envs.base import BaseEnv
from core.utils.typing import ActionType, ObsType
from experiments.tuner import BasicTuner


class MockAgent(BaseAgent):

    def act(self, obs: ObsType) -> ActionType:
        return self._action_space.sample()
    
    def learn(self, obs: ObsType, action: ActionType, reward: float, next_obs: ObsType, terminal: bool) -> None:
        pass

    def turn_eval_mode_on(self): ...
    def turn_eval_mode_off(self): ...

class MockEpisodicEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self._action_space = spaces.Discrete(2)
        self._obs_space = spaces.Box(low=0, high=1, shape=(2,))
        self.termination_prob = 0.05
    
    def reset(self):
        return self._obs_space.sample(), {}
    
    def _act(self, action: ActionType): ...

    def _observe(self):
        terminal = False
        if random() < self.termination_prob:
            terminal = True
        return self._obs_space.sample(), random(), terminal, False, {}

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

#FIX: Pytest get stuck when I'm running multiprocessing tests on pooling (i.e., increasing the number of workers to 2 or more)
@pytest.mark.parametrize("args", [{
    'config_file': Path('experiments/configs/tests/algo/online/mock_algo.yaml'),
    'gpu': False,
    'verbose': logging.INFO,
    'workers': 1, 
    'run': 1
}, {
    'config_file': Path('experiments/configs/tests/algo/online/mock_algo.yaml'),
    'gpu': False,
    'verbose': logging.INFO,
    'workers': 1,
    'run': 5
}])
def test_execution(args):
    args = SimpleNamespace(**args)    
    config_file = args.config_file.absolute()
    tuner = BasicTuner.create_by_config_file(config_file, verbose=args.verbose, workers=args.workers, gpu=args.gpu, run=args.run)
    best_cfg = tuner.tune()
    shutil.rmtree(tuner.log_dir)

def test_metrics_storage_capability():
    """Test the major parameters of the algorithm.
    """
    pass

