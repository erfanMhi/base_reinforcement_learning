from typing import List

import numpy as np

from core.utils.exploration.base import BaseExploration
from core.utils.typing import ActionType

class EpsilonGreedyExploration(BaseExploration):

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    #TODO: to be replaced with action distribution class 
    def determine_action(self, action_distro: List[float]) -> ActionType:
        if np.random.rand() < self._epsilon:
            return np.random.choice(len(action_distro))
        else:
            return np.argmax(action_distro)