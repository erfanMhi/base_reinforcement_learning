import random
import logging
from typing import Any, List, Tuple 

import numpy as np
import torch


from core.utils.typing import ActionType, ObsType, INTType, FLOATType
from core.utils.replay_buffers.base import BaseReplayBuffer

logger = logging.getLogger(__name__)

def _sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class EfficientReplayBuffer(BaseReplayBuffer):
    def __init__(self, size: int, frame_history_len: int = 1):
        super().__init__(size)
        self._frame_history_len = frame_history_len
        self._initialize()

    def reinitialize(self):
        self._initialize()
    
    def _initialize(self):
        self._next_idx      = 0
        self._num_in_buffer = 0

        self._obs      = None
        self._action   = None
        self._reward   = None
        self._done     = None

    def sample(self, batch_size: int) -> Tuple[List[ObsType], List[ActionType], List[float], List[ObsType], List[float]]:
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
        assert self.can_sample(batch_size)
        idxes = _sample_n_unique(lambda: random.randint(0, self._num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        assert self._num_in_buffer > 0
        return self._encode_observation((self._next_idx - 1) % self._size)

    def _encode_observation(self, idx: int) -> ObsType:
        """adding the previous frames to the current observation to create a markov state

        Args:
            idx (int): index of the required observation in the replay buffer

        Returns:
            ObsType: the encoded observation
        """
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self._frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest observation
        if len(self._obs.shape) == 2: #TODO: low dimensional frame add is not supported
            return self._obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self._num_in_buffer != self._size:
            start_idx = 0
        # if there aren't enough frames in the current episode, skip it and compute the missing frames
        for idx in range(start_idx, end_idx - 1):
            if self._done[idx % self._size]:
                start_idx = idx + 1
        missing_context = self._frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self._obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self._obs[idx % self._size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self._obs.shape[1], self._obs.shape[2]
            return self._obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def _encode_sample(self, idxes: List[int]) -> Tuple[List[ObsType], List[ActionType], List[float], List[ObsType], List[float]]:
        """
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        """
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self._action[idxes]
        rew_batch      = self._reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self._done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
    
    def store_obs(self, obs: ObsType) -> int:

        if self._obs is None:
            obs_shape, obs_type = self._compute_obs_shape_and_type(obs)
            logger.debug('observation shape determined: %s', obs_shape)
            logger.debug('observation type determined: %s', obs_type)
            self._obs = np.empty((self._size,) + obs_shape, dtype=obs_type)

        self._obs[self._next_idx] = obs

        ret = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._size
        self._num_in_buffer = min(self._size, self._num_in_buffer + 1)

        return ret

    @classmethod
    def _compute_obs_shape_and_type(cls, obs: ObsType) -> Tuple[Tuple[int, ...], Any]:
        """Compute the shape and type of the observation.
        Parameters
        ----------
        obs: np.array
            The observation.
        Returns
        -------
        shape: Tuple[int, ...]
            The shape of the observation, including the frame history.
        """
        if cls._is_image(obs):
            return obs.shape, np.uint8
        elif isinstance(obs, (np.ndarray, torch.Tensor)):
            return obs.shape, obs.dtype
        elif isinstance(obs, list):
            element = obs[0]
            if isinstance(element, int):
                return len(obs), np.int32
            elif isinstance(element, float):
                return len(obs), np.float32
            else:
                raise ValueError("Unsupported observation type: {}".format(type(element)))
        else:
            raise ValueError("Unsupported observation type: {}".format(type(obs)))

    @staticmethod
    def _is_image(obs: ObsType) -> bool:
        if isinstance(obs, (np.ndarray, torch.Tensor)):
            return len(obs.shape) == 3
        return False

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
        if self._action is None:
            action_shape, action_type = self._compute_action_shape_and_type(action)
            logger.debug('action shape determined: %s', action_shape)
            logger.debug('action type determined: %s', action_type)
            self._action   = np.empty((self._size,) + action_shape, dtype=action_type)
            self._reward   = np.empty((self._size,), dtype=np.float32)
            self._done     = np.empty((self._size,), dtype=np.bool)
        self._action[idx] = action
        self._reward[idx] = reward
        self._done[idx]   = done

    @staticmethod
    def _compute_action_shape_and_type(action: ActionType) -> Tuple[Tuple[int, ...], Any]:
        if isinstance(action, np.ndarray):
            return action.shape, action.dtype
        elif isinstance(action, list):
            element = action[0]
            if isinstance(element, (int, np.int64, np.int32)):
                return len(action), np.int32
            elif isinstance(element, float):
                return len(action), np.float32
            else:
                raise ValueError("Unsupported action type: {}".format(type(element)))
        elif isinstance(action, (int, np.int64, np.int32)):
            return (), np.int32
        elif isinstance(action, float):
            return (), np.float32
        else:
            raise ValueError("Unsupported action type: {}".format(type(action)))

    def create_eval_replay_buffer(self) -> BaseReplayBuffer:
        """Create a replay buffer for evaluation.
        Returns
        -------
        replay_buffer: ReplayBuffer
            The replay buffer for evaluation.
        """

        return EfficientReplayBuffer(self._frame_history_len, self._frame_history_len)

    