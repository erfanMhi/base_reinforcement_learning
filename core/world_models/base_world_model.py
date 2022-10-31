import abc
from typing import Dict, Tuple

from core.utils.typing import ActionType, ObsType, RewardType

class WorldModel(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def transit(self, obs: ObsType, action: ActionType) -> Tuple[RewardType, ObsType]: ...

    @abc.abstractmethod
    def get_transit_prob(self, obs: ObsType, action: ActionType) -> Dict[ObsType, float]: ...
    