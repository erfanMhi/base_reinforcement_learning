from abc import ABC, abstractmethod
from typing import List

from core.utils.typing import ActionType

class BaseExploration(ABC):

    #TODO: to be replaced with action distribution class 
    @abstractmethod
    def determine_action(self, action_distro: List[float]) -> ActionType:
        pass