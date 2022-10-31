from typing import (
    Dict,
    Any
)
from dataclasses import dataclass, field

from core.utils.configs import BaseConfig

@dataclass
class DQNConfig(BaseConfig):

    discount: float = 0.99

    # agent parameters
    exploration: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'epislon-greedy',
        'epsilon': 0.1
    })
        
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'adam',
        'lr': 1e-4,
        'momentum': 0.9
    })

    target_net: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'simple',
        'update_frequency': 1024
    })

    model: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'mlp',
        'hidden_layers': [32, 32],
        'activation': 'relu'
    })

    loss: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'mse'
    })

    # replay buffer parameters
    memory_size: int = 100000
    batch_size: int = 32

    # training parameters
    update_per_step: int = 1
    max_steps: int = 100000

    # log parameters
    log_interval: int = 1
    returns_queue_size: int = 100 # used to calculate the average return
        