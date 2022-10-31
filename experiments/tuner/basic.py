from torch import multiprocessing
import logging
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Generator,
    Union
)
from pathlib import Path
from copy import deepcopy

import numpy as np
from core.algos.base import BaseAlgo

from core.utils.configs import BaseConfig
from core.utils.io import load_yaml
from core.utils.typing import ScalarType
from core.utils.util import iterate_nested_dict
from experiments.tuner.base import BaseTuner

BEST_CONFIG_FILE_NAME = 'best_config.yaml'

logger = logging.getLogger(__name__)

class BasicTuner(BaseTuner):
    """
    This class is responsible for reading a YAML file and finding the right parameter setting
    and random seed for the config class (based on the given id).
    Some important notes to consider:
        --- The parameters in config dictionary/file should be already defined by 
            the config class: in order to add a new parameter to the config dictionary/file
            you have to add it to the config class first.
        --- This class is parameter-structure agnostic, meaning that it doesn't have any approach
            to validate the nested parameters in the YAML file. The classes that use those parameters 
            are responsible for validating them.
        --- Id starts from '0'
        --- YAML file key names should be as same as the config class field names
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._evaluation_metrics = None

    def _tune(self) -> BaseConfig:
        configs = self._get_config_generator()
        self._evaluation_metrics = self._parallelize_algo_runs(configs)
        best_cfg = self._get_best_config(self._evaluation_metrics)
        logger.info("Best config run: {}".format(best_cfg.seed))
        logger.info("Best config param_setting: {}".format(best_cfg.param_setting))
        self._save_best_config_results(best_cfg)
        return best_cfg

    def _parallelize_algo_runs(self, configs: Union[Generator, List[BaseConfig]]) -> List[Tuple[ScalarType]]:
        if self._workers == 1:
            evaluation_metrics = list(map(self._make_and_run_algo, configs))
        else:
            with multiprocessing.Pool(self._workers) as p:
                evaluation_metrics = p.map(self._make_and_run_algo, configs)
        return evaluation_metrics

    def _get_config_generator(self) -> Generator[BaseConfig, None, None]:
        for run in range(self._run):
            for param_setting in range(self._sweep_size):
                id_ = self._get_id(run, param_setting)
                cfg = self._get_config(id_)
                yield cfg

    def _get_best_config(self, evaluation_metrics: List[Tuple[ScalarType]]) -> BaseConfig:
        evaluation_metrics_id_combined = zip(evaluation_metrics, range(len(evaluation_metrics)))
        sorted_evaluation_metrics_id_combined = sorted(evaluation_metrics_id_combined, key=lambda x: x[0], reverse=True)
        best_config_id = sorted_evaluation_metrics_id_combined[0][1]
        return self._get_config(best_config_id)

    def _save_best_config_results(self, best_cfg: BaseConfig) -> None:
        best_cfg.save_config_to(self._log_dir/BEST_CONFIG_FILE_NAME)
        logger.info('Best config is saved to {}'.format(self._log_dir/BEST_CONFIG_FILE_NAME))

    def _get_config(self, id_: int) -> BaseConfig:
        meta_params = self._build_meta_params()
        algo_params = self._build_algo_params(id_)
        seed = self._generate_seed(id_)
        param_setting = self._generate_param_setting(id_)
        
        config_args = self._create_config_args(id_, seed, param_setting, **meta_params, **algo_params)

        config = self._config_class(**config_args)
        return config

    def _build_algo_params(self, id_: int) -> None:
        algo_params = self._extract_algo_params()
        algo_params = deepcopy(algo_params)
        sweep_interval = 1
        for dic, param, value in iterate_nested_dict(algo_params):
            if isinstance(value, dict) and self._is_search_dict(value):
                self._validate_search_dict(value)
                search_params = self._retrieve_search_params_from_search_dict(value)
                num_params = len(search_params)
                chosen_param_index = (id_ // sweep_interval) % num_params
                dic[param] = search_params[chosen_param_index]
                sweep_interval *= num_params
        return algo_params

    def _build_meta_params(self) -> None:
        meta_params = self._extract_meta_params()
        meta_params = deepcopy(meta_params)
        return meta_params

    def _get_id(self, run: int, param_setting: int) -> int:
        return run * self._sweep_size + param_setting

    def _generate_seed(self, id_: int) -> int:
        return id_ // self._sweep_size

    def _generate_param_setting(self, id_: int) -> int:
        return id_ % self._sweep_size
    
    def _retrieve_search_params_from_search_dict(self, search_dict: Dict[str, List[Any]]) -> List[Any]:
        # return search_dict[PARAMS_SEARCH_UNIFIED_ID]
        return list(search_dict.values())[0]

    @staticmethod
    def _validate_search_dict(search_dict: Dict[str, List[Any]]) -> None:
        """Enforcing search dictionaries to have a single key/value pair"""
        if len(search_dict) != 1:
            raise ValueError('There are more than one search key in the search dictionary')

    @property
    def evaluation_metrics(self) -> List[Tuple[ScalarType]]:
        return self._evaluation_metrics
        
