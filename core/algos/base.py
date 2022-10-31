from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
import logging
import multiprocessing
import random
import shutil
import time
from typing import Tuple, Type

import torch
import numpy as np
from pprint import pformat

from core.utils.configs.base import BaseConfig
from core.utils.recorders.tensorboard import TensorboardRecorder
from core.utils.typing import ScalarType
from core.utils.util import get_current_datetime, retrieve_subclass

LOG_FILE_EXTENSION = '.log'
RUNNING_FINISHED_SIGNAL = 'finished'
ROOT_FILE_NAME = 'core'

logger = logging.getLogger(__name__)

class BaseAlgo(ABC):

    def __init__(self, cfg: BaseConfig) -> None:
        self._cfg = cfg
        self._log_file = self._get_log_file_path()
        self._seed = self._cfg.seed
        self._deterministic = self._cfg.deterministic
        self._id = self._get_id()

        self._set_global_seeds(self._seed)
        self._make_cuda_deterministic(self._deterministic)

    def _get_log_file_path(self) -> str:
        log_file_name = get_current_datetime()+LOG_FILE_EXTENSION
        return self._cfg.log_dir/log_file_name
    
    def _get_id(self) -> str:
        return multiprocessing.current_process().name + '()'.format(str(self._cfg.id))

    def _set_global_seeds(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _make_cuda_deterministic(self, flag: bool) -> None: #TODO: check if this works
        if flag:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False

    def run(self) -> Tuple[ScalarType]: #TODO: add pytorch profiler for profiling the CPU, GPU, and memory usage https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html?highlight=tensorboard
        """Runs the algorithm and returns the evaluation metrics
        Features:
            - Checks if the results already exist
            - It takes away the burden of saving and retrieving the metrics from the existing logs
            - Wall clock time and CPU time are automatically added to the metrics
        Returns:
            Tuple[ScalarType]: the metrics ordered by their priority for evluation
        """
        if self._results_exist():
            logger.info('Results for run {} already exist'.format(self._id))
            metrics = self._retrieve_metrics()
            logger.info(f'Retrieved metrics for run {self._id}: {pformat(metrics)}')
        else:
            self._prepare_log_dir()
            self._start_recording_time()
            logger.info(f'Running run {self._id}')
            metrics = self._run()
            self._end_recording_time()
            logger.info(f'Run {self._id} finished in {self._wall_time}s wall time and {self._cpu_time}s cpu time')
            logger.info(f'Run {self._id} finished with metrics: {pformat(metrics)}')
            metrics = {**metrics, 'wall_time': self._wall_time, 'cpu_time': self._cpu_time}
            self._save_metrics(metrics)
        metrics = tuple(metrics.values())
        return metrics

    def _results_exist(self) -> bool:
        if not self._cfg.log_dir.exists():
            return False
        for path in self._cfg.log_dir.iterdir():
            if path.is_file() and path.name.endswith(LOG_FILE_EXTENSION) and self._is_experiment_finished(path):
                return True
        shutil.rmtree(self._cfg.log_dir)
        return False
    
    def _is_experiment_finished(self, path) -> bool:
        with open(path, 'r') as f:
            lines = f.read()
            if RUNNING_FINISHED_SIGNAL in lines:
                return True

    def _prepare_log_dir(self) -> None:
        self._try_make_log_dir()
        self._initialize_recorder()
        self._initialize_logger()
        self._cfg.save_config()

    def _initialize_recorder(self):
        self._cfg.recorder = TensorboardRecorder(self._cfg.log_dir, self._cfg.verbose)
    
    def _initialize_logger(self):
        logger = logging.getLogger(ROOT_FILE_NAME)
        logger.setLevel(self._cfg.verbose)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(self._log_file)
        handler.setLevel(self._cfg.verbose)
        handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(handler)

    def _try_make_log_dir(self):
        if not self._cfg.log_dir.exists():
            self._cfg.log_dir.mkdir(parents=True)

    def _start_recording_time(self) -> None:
        self._wall_time = time.time()
        self._cpu_time = time.process_time()

    def _end_recording_time(self) -> None:
        self._wall_time = time.time() - self._wall_time
        self._cpu_time = time.process_time() - self._cpu_time

    @abstractmethod
    def _run(self) -> OrderedDict: ...

    def _retrieve_metrics(self) -> Tuple[ScalarType]:
        keys = self._get_metrics_keys()
        metrics = OrderedDict()
        for key in keys:
            value = TensorboardRecorder.retrieve_metric_value(self._cfg.log_dir, key)
            metrics[key] = value
        return metrics

    @abstractmethod
    def _get_metrics_keys(self) -> Tuple[ScalarType]:  #TODO: find a replacement for this approach
        return ['wall_time', 'cpu_time']

    def _save_metrics(self, metrics: Tuple[ScalarType]) -> None:
        hparams = self._cfg.get_flatten_params()
        self._cfg.recorder.add_hparams(hparams, metrics, logging.INFO)

    @classmethod
    def retrieve_algo(cls, name: str) -> 'BaseAlgo':
        return retrieve_subclass(cls, name)
