from abc import ABC, abstractmethod
import logging
import os
import sys
from pathlib import Path
from typing import (
    Any, 
    Dict, 
    Generator, 
    List,
    Tuple,
    Union
)

import torch
import numpy as np

from core.algos.base import BaseAlgo
from core.utils.configs import BaseConfig
from core.utils.io import load_yaml
from core.utils.util import get_current_datetime, iterate_nested_dict
from experiments.sweeper.search import GridSearch, UniformSearch
from core.utils.typing import ScalarType

# This list is only targeted toward search keys used in YAML files.
PARAMS_SEARCH_CLASSES = {
    'grid_search': GridSearch, 
    'uniform_search': UniformSearch,
}

# The same logic as grid-search. Used only for the implementaiton purposes. It shouldn't be included in the YAML files.
PARAMS_SEARCH_UNIFIED_ID = 'param_search' 

CONFIG_CLASS_ID = 'config_class'
META_PARAMS_ID = 'meta-params'
ALGO_PARAMS_ID = 'algo-params'
FIRST_LEVEL_KEYS = [CONFIG_CLASS_ID, META_PARAMS_ID, ALGO_PARAMS_ID]
SEARCH_KEY_INDEX = 0

LOG_FILE_EXTENSION = '.log'

logger = logging.getLogger(__name__)

class BaseTuner(ABC):

    def __init__(
        self, cfg_params: Dict[str, Any], verbose=logging.INFO,
        run: int = 1, workers: int = -1, seed: int = 0, gpu: bool = False) -> None:
        """_summary_

        Args:
            cfg_params (Dict[str, Any]): dictionary of parameters for the tuner
            verbose (_type_, optional): logging level: INFO and DEBUG are supported for now. Defaults to logging.INFO.
            run (int, optional): number of times that each algorithm needs to be evaluated. Defaults to 1.
            workers (int, optional): number of workers used to run the experiments. -1 means that the number of runs are going to be automatically determined. Defaults to -1.
            seed (int, optional): the seed used to only determine the hyperparameters of the tuner class. Defaults to 0.
        """
        self._cfg_params = cfg_params
        self._gpu = gpu
        self._verbose = verbose
        self._run = run
        self._workers = workers
        self._seed = seed
        self._log_dir = self._extract_log_dir()
        self._try_make_log_dir()
        self._log_file = self._get_log_file()
        self._random_generator = np.random.default_rng(self._seed)
        self._search_keys = set(PARAMS_SEARCH_CLASSES.keys())
        self._config_class = self._extract_config_class()

        self._validate_first_level_keys()
        self._initilize_logger()
        self._preprocess_config()
    
    def _try_make_log_dir(self):
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _extract_log_dir(self) -> Path:
        log_dir = self._cfg_params['meta-params']['log_dir']
        log_dir = Path(log_dir).absolute()
        return log_dir

    def _get_log_file(self) -> Path:
        log_file_name = get_current_datetime()+LOG_FILE_EXTENSION
        log_file = self._log_dir/log_file_name
        return log_file

    def _extract_config_class(self) -> BaseConfig:
        config_class_name = self._cfg_params[CONFIG_CLASS_ID]
        config_class = BaseConfig.retrieve_config(config_class_name)
        return config_class

    def _validate_first_level_keys(self) -> None:
        first_level_keys_set = set(FIRST_LEVEL_KEYS)
        cfg_keys_set = set(self._cfg_params.keys())
        if first_level_keys_set != cfg_keys_set:
            raise TypeError(f'The following first level keys should always be included in the config file: {first_level_keys_set}. But, the following keys are given: {cfg_keys_set}')
    
    def _preprocess_config(self) -> None:
        """This function converts search parameters to a grid-search like search parameters. 
        """
        self._sweep_size = 1
        algo_params = self._extract_algo_params()
        for dic, param, value in iterate_nested_dict(algo_params):
            if self._is_search_key(param):
                self._validate_search_dict(dic)
                dic[param] = self._get_unified_search_params(param, value)
                self._sweep_size *= len(dic[param])

    def _get_unified_search_params(self, search_key: str, raw_search_params: List[Any]) -> List[Any]: 
        search_class = PARAMS_SEARCH_CLASSES[search_key]
        search_obj = search_class(raw_search_params)
        search_params = search_obj.get_search_params(self._random_generator)
        return search_params

    def _initilize_logger(self):
        logger = logging.getLogger('experiments.tuner')
        logger.setLevel(self._verbose)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(self._verbose)
        logger.addHandler(stdout_handler)
        file_handler = logging.FileHandler(self._log_file)
        file_handler.setLevel(self._verbose)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    @abstractmethod
    def _tune(self) -> BaseConfig:
        pass
    
    def tune(self) -> BaseConfig:
        self._workers = self._determine_workers_num()
        return self._tune()

    def _determine_workers_num(self) -> int: #TODO: compute this value by running a small set of workers first and then use the result to compute the optimal number of workers
        if self._workers == -1:
            if self._gpu:
                return min(os.cpu_count(), torch.cuda.device_count())
            else:
                return os.cpu_count()
        return self._workers

    def _create_config_args(self, id_: int, seed: int, param_setting: int, **kwargs) -> Dict[str, Any]:
        config_args = {
            'id': id_, 'seed': seed, 'param_setting': param_setting, 
            'verbose': self._verbose,
            CONFIG_CLASS_ID: self._config_class.__name__,
            **kwargs
        }
        return config_args


    def _make_and_run_algo(self, cfg: BaseConfig) -> Tuple[ScalarType]: 
        cfg.device = self._assign_device()
        algo_class = BaseAlgo.retrieve_algo(cfg.algo_class) 
        algo = algo_class(cfg)
        return algo.run()

    def _assign_device(self) -> str: #TODO: maybe implement a procedure to find Free GPUs: https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/6 & https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
        if self._gpu:
            return 'cuda'
        else:
            return 'cpu'

    def _is_search_dict(self, dic: Dict[str, Any]) -> bool:
        return any([search_key in dic for search_key in self._search_keys])

    def _is_search_key(self, search_key: str) -> bool:
        return search_key in self._search_keys
    
    def _extract_algo_params(self) -> Dict[str, Any]:
        return self._cfg_params[ALGO_PARAMS_ID]

    def _extract_meta_params(self) -> Dict[str, Any]:
        return self._cfg_params[META_PARAMS_ID] 
    
    @staticmethod
    def _load_config(config_file: Path) -> Dict[str, any]:
        config_params = load_yaml(config_file)
        return config_params

    @classmethod
    def create_by_config_file(cls, config_file: Path, **kwargs) -> 'BaseTuner':
        config_dict = cls._load_config(config_file)
        config_dict['meta-params']['config_file'] = config_file
        cfg = cls(config_dict, **kwargs)
        return cfg

    @property
    def log_dir(self) -> Path:
        return self._log_dir