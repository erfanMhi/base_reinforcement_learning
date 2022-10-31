import logging
from collections import OrderedDict
from random import random
from typing import Tuple

import torch
import numpy as np

from core.algos.base import BaseAlgo
from core.agents import BaseAgent
from core.envs import BaseEnv
from core.utils.configs.base import BaseConfig
from core.utils.typing import ScalarType

EPISODE_RETURNS_KEY = 'returns'

logger = logging.getLogger(__name__)

class EfficientQueue:

    def __init__(self, size) -> None:
        self._items = np.zeros(size)
        self._size = size
        self._pointer = 0

    def enqueue(self, item) -> None:
        self._items[self._pointer] = item
        self._pointer = (self._pointer + 1) % self._size

    def average(self) -> float:
        return np.mean(self._items)

class OnlineAlgo(BaseAlgo):

    def __init__(self, cfg: BaseConfig) -> None:
        super().__init__(cfg)
        self._cfg = cfg
        self._max_steps = self._cfg.max_steps
        self._log_interval = self._cfg.log_interval
        self._returns_queue_size = self._cfg.returns_queue_size

        # logging parameters
        self._returns_queue = EfficientQueue(self._returns_queue_size) #TODO: only work for episodic environments for now

        self._agent_class = BaseAgent.retrieve_agent(self._cfg.agent_class)
        self._env_class = BaseEnv.retrieve_env(self._cfg.env_class)

        self._step = 0
        self._episode = 0

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        assert self._max_steps > 0, 'The maximum number of steps must be positive'
        assert self._log_interval > 0, 'The logging interval must be positive'
        assert self._returns_queue_size > 0, 'The returns queue size must be positive'
        assert self._max_steps % self._log_interval == 0, 'The maximum number of steps must be a multiple of the logging interval'

    def _run(self) -> Tuple[ScalarType]:

        self._env = self._env_class(self._cfg)
        # add the environment info to the global configs
        self._cfg.obs_space = self._env.observation_space
        self._cfg.action_space = self._env.action_space
        self._agent = self._agent_class(self._cfg)

        # populate the returns queue: used only for logging purposes
        self._populate_returns_queue()

        terminated = True
        truncated = True #TODO: add truncated functionality: requires edition of the replay buffer
        auc = 0
        total_reward = 0
        for step in range(self._max_steps):
            #NOTE: this implementation doesn't let the agent store the last observation in the replay buffer
            if step % self._log_interval == 0:
                self._record_returns()
                auc += self._returns_queue.average()

            if terminated:
                obs, _ = self._env.reset()
                if step != 0: 
                    self._add_return(episode_return)
                    self._episode += 1
                    logger.debug(f'(Training Phase) Episode {self._episode} with return {episode_return}')
                episode_return = 0
                terminated = False
                truncated = False

            # single agent-environment interaction
            action = self._agent.act(obs)
            next_obs, reward, terminated, truncated, _ = self._env.step(action)
            self._agent.learn(obs, action, reward, next_obs, terminated) #NOTE: Termination/truncation distinction can result in bad learning when we need to stack multiple frames in an environment (only if the environment actually uses the truncation flag)
            obs = next_obs

            # update logging parameters
            episode_return += reward
            total_reward += reward
            self._step += 1
    
        if self._step % self._log_interval == 0:
            self._record_returns()
            auc += self._returns_queue.average()

        metrics = OrderedDict({
            'auc': auc,
            'total_reward': total_reward,
            'episodes': self._episode,
        })

        return metrics


    def _populate_returns_queue(self) -> None: #TODO: make this part deterministic and independent samples from other parts of the code
        """This method is used to make sure we have accurate estimate of the return for logging purposes in the beginning of training"""
        self._agent.turn_eval_mode_on()
        episodes = self._returns_queue_size
        for episode in range(episodes):
            obs, _ = self._env.reset()
            terminal = False
            episode_return = 0
            while not terminal:
                action = self._agent.act(obs)
                next_obs, reward, terminal, _, _ = self._env.step(action)
                self._agent.learn(obs, action, reward, next_obs, terminal)
                episode_return += reward
                obs = next_obs
            logger.debug(f'(Evaluation Phase) Episode {episode} with return {episode_return}')
            self._add_return(episode_return)
        self._agent.turn_eval_mode_off()
    
    def _record_returns(self) -> None:
        self._cfg.recorder.add_scalar(EPISODE_RETURNS_KEY, self._returns_queue.average(), self._step, logging.INFO)

    def _add_return(self, episode_return):
        self._returns_queue.enqueue(episode_return)

    def _get_metrics_keys(self) -> Tuple[ScalarType]: #TODO: find a replacement for this approach
        metrics_keys = super()._get_metrics_keys()
        return ['auc', 'total_reward', 'episodes'] + metrics_keys


