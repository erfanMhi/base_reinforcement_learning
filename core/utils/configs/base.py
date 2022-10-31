from copy import deepcopy
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import OrderedDict, Union
from shutil import copyfile

import torch
import numpy as np

from core.utils.util import retrieve_subclass
from core.utils.io import save_yaml

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = 'config.yaml'
BEST_CONFIG_FILE_NAME = 'best_config.yaml'

def _flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@dataclass
class BaseConfig(ABC):

    id: int
    param_setting: int
    seed: int
  
    config_class: str
    config_file: Union[Path, str]
    agent_class: str
    env_class: str
    algo_class: str

    log_dir: Union[Path, str]
    verbose: int
    device: str = 'cpu'
    deterministic: bool = True

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self.log_dir = self.log_dir.absolute()
        self.log_dir = Path(self.log_dir)/str(self.seed)/str(self.param_setting)

    def save_config(self) -> None:
        save_yaml(self.log_dir/CONFIG_FILE_NAME, self.get_params(self.__dict__))

    def save_config_to(self, destination: Path) -> None:
        save_yaml(destination, self.get_params(self.__dict__))
        
    def copy_original_config_yaml(self, destination: Path) -> None:
        source = Path(self.config_file)
        copyfile(source, destination)

    def get_flatten_params(self, tensorboard=True) -> dict:
        flatten_params = _flatten(self.__dict__)
        flatten_params = self.get_params(flatten_params, tensorboard)
        return flatten_params

    @staticmethod
    def get_params(params: dict, tensorboard=False) -> dict:
        params_ = {}
        dict_items = tuple(params.items())
        for key, value in dict_items:
            if isinstance(value, Path):
                params_[key] = str(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                if tensorboard:
                    params_[key] = torch.tensor(value)
                else:
                    params_[key] = value
            elif isinstance(value, (int, float, str, bool)):
                params_[key] = value
            elif isinstance(value, dict):
                params_[key] = BaseConfig.get_params(value)
        return params_

    @property
    def root_log_dir(self) -> Path:
        return self.log_dir.parent.parent

    @classmethod
    def retrieve_config(cls, name: str) -> 'BaseConfig':
        return retrieve_subclass(cls, name)


    
  
