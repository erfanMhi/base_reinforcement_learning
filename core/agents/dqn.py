from copy import deepcopy
import logging
from symbol import parameters
from typing import List

import torch
from torch import nn

from core.agents.base import BaseAgent
from core.utils.replay_buffers.efficient import EfficientReplayBuffer
from core.utils.typing import ActionType, TerminalType, ObsType, RewardType 
from core.utils.util import extract_name_and_args

logger = logging.getLogger(__name__)

class TargetNet(nn.Module): # TODO: turn this into a baseclass, with each class having different implementation for try_update function
    def __init__(self, net: nn.Module, name: str, update_frequency: int) -> None:
        super().__init__()
        self._name = name
        self._update_frequency = update_frequency
        self._net = deepcopy(net)

    def forward(self, x):
        return self._net(x)

    def try_update(self, net: nn.Module, step_count: int) -> bool:
        if step_count % self._update_frequency != 0:
            return False

        if self._name == 'discrete':
            self._net.load_state_dict(net.state_dict())
        else:
            raise ValueError(f'Unknown target net type: {self._name}')
        return True

class DQNAgent(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._update_per_step = self._cfg.update_per_step
        self._discount = self._cfg.discount
        self._batch_size = self._cfg.batch_size
        self._log_interval = self._cfg.log_interval
        #TODO: to be updated with the global replay buffer shared between workers in the future
        self._local_replay_buffer = self._training_replay_buffer = self._create_replay_buffer(self._cfg.memory_size)
        self._current_obs_idx = None # used to indicate where the last observation is stored in the replay buffer

        self._behavior_net = self._create_network(**self._cfg.model).to(self._device)
        self._target_net = TargetNet(self._behavior_net, **self._cfg.target_net).to(self._device)
        self._record_network_graphs()
        parameters = self._behavior_net.parameters()
        self._optimizer = self._create_optimizer(parameters, **self._cfg.optimizer)
        self._loss_func = self._create_loss(**self._cfg.loss)
        self._exploration = self._create_exploration(**self._cfg.exploration)

        self._step_count = 0
        self._record_weights()

    @staticmethod
    def _create_replay_buffer(memory_size: int) -> None: #TODO: write a factory function for replay buffer and also use non-efficient replay buffer for the gym-like environments that store both current and next observations. At this moment, this replay buffer results in a slight bug when truncation happens
        return EfficientReplayBuffer(memory_size)

    def _record_network_graphs(self):
        input_to_model = torch.from_numpy(self._obs_space.sample()[None, :]).float().to(self._device)
        self._cfg.recorder.add_graph(self._behavior_net, input_to_model, logging.DEBUG)
        self._cfg.recorder.add_graph(self._target_net, input_to_model, logging.DEBUG)

    def _record_weights(self):
        if self._step_count % self._log_interval == 0:
            for tag, values in self._behavior_net.named_parameters(prefix='behavior_net'):
                self._cfg.recorder.add_histogram(tag, values, self._step_count, logging.DEBUG)
            for tag, values in self._target_net.named_parameters(prefix='target_net'):
                self._cfg.recorder.add_histogram(tag, values, self._step_count, logging.DEBUG)

    def act(self, obs: ObsType) -> ActionType:
        self._current_obs_idx = self._local_replay_buffer.store_obs(obs)
        current_obs = self._local_replay_buffer.encode_recent_observation()
        current_obs = torch.from_numpy(current_obs).float().to(self._device)
        q_values = self._get_q_vals_no_grad(current_obs).numpy()
        action = self._exploration.determine_action(q_values)
        return action
    
    @torch.no_grad()
    def _get_q_vals_no_grad(self, obs: ObsType) -> List[float]:
        q_values = self._behavior_net(obs)
        return q_values

    #TODO: this function should take its replay buffer inputs from localworkers instead of the algorithm
    def learn(self, obs: ObsType, action: ActionType, reward: RewardType, next_obs: ObsType, terminal: TerminalType) -> None:
        
        self._local_replay_buffer.store_effect(self._current_obs_idx, action, reward, terminal)
        if self._local_replay_buffer.can_sample(self._batch_size):
            for _ in range(self._update_per_step):
                self._single_training_update()

        self._target_net.try_update(self._behavior_net, self._step_count)   
        self._step_count += 1
        self._record_weights()

    def _single_training_update(self):
        obs_batch, act_batch, rew_batch, next_obs_batch, terminal_batch = self._local_replay_buffer.sample(self._batch_size)
        #TODO: this should be done in more graceful way
        obs_batch = torch.from_numpy(obs_batch).float().to(self._device)
        act_batch = torch.from_numpy(act_batch).long().to(self._device)
        rew_batch = torch.from_numpy(rew_batch).float().to(self._device)
        next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self._device)
        terminal_batch = torch.from_numpy(terminal_batch).float().to(self._device)

        q_values = self._behavior_net(obs_batch)
        q_values = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)
        target_q_values = self._compute_targets(next_obs_batch, rew_batch, terminal_batch)
        loss = self._loss_func(q_values, target_q_values)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        self._cfg.recorder.add_scalar('loss', loss.item(), self._step_count, logging.DEBUG)

    @torch.no_grad()
    def _compute_targets(self, next_obs_batch: List[ObsType], rew_batch: List[RewardType], terminal_batch: List[TerminalType]) -> torch.Tensor:
        next_q_values = self._target_net(next_obs_batch)
        next_q_values = next_q_values.max(1)[0]
        target_q_values = rew_batch + self._discount * next_q_values * (1 - terminal_batch)
        return target_q_values

    def turn_eval_mode_on(self):
        self._training_replay_buffer = self._local_replay_buffer
        self._training_step_count = self._step_count
        self._step_count = 0
        self._local_replay_buffer = self._local_replay_buffer.create_eval_replay_buffer()
        self._behavior_net.eval()
        self._target_net.eval()
        self._update_per_step = 0

    def turn_eval_mode_off(self):
        self._local_replay_buffer = self._training_replay_buffer
        self._update_per_step = self._cfg.update_per_step
        self._step_count = self._training_step_count
        self._behavior_net.train()
        self._target_net.train()