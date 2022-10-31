from abc import ABC, abstractmethod
import random
from typing import List, Tuple 

import numpy as np

from core.utils.typing import ActionType, ObsType

class BaseReplayBuffer(ABC):
    """This design for the replay buffer gives more responsibility to the agent. The agent is responsible for making sure that it stores_effect after storing observation 
    """
    def __init__(self, size: int, **kwargs):
        self._size = size

    def can_sample(self, batch_size: int) -> bool:
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self._num_in_buffer

    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[List[ObsType], List[ActionType], List[float], List[ObsType], List[int]]:
        """Sample `batch_size` different transitions.
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        pass

    @abstractmethod
    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        pass

    @abstractmethod
    def store_obs(self, obs: ObsType) -> int:
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        pass

    @abstractmethod
    def store_effect(self, idx: int, action: ActionType, reward: float, done: bool) -> None:
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        pass