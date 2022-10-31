import os
import sys
import argparse
import random
import logging
from pathlib import Path
from typing import (
    Any,
    List,
)

import torch
import numpy as np

from core.algos import BaseAlgo
from core.utils.configs.base import BaseConfig
from core.utils.exceptions import GPUNotAvailable
from experiments.sweeper import BasicSweeper
from core.utils.typing import GPUID, ArgsObject
from core.utils.recorders import TensorboardRecorder

LOG_FILE_NAME = 'logs.log'
logger = logging.getLogger(__name__)
    
def parse_args(args: List[Any]) -> ArgsObject:

    parser = argparse.ArgumentParser(description="The main file for running experiments")

    parser.add_argument('--config-file', dest='config_file', type=Path, help='Expect a json file describing the fixed and sweeping parameters')

    parser.add_argument('--id', default=0, type=int, help='Identifies param_setting, seed, and parameters configuration')

    parser.add_argument('--gpu', action='store_true', help="Use GPU: if not specified, use CPU")

    parser.add_argument('--verbose', dest='verbose', choices=[logging.INFO, logging.DEBUG], default=logging.INFO, type=int)

    args = parser.parse_args(args)

    return args

def _select_device(gpu_id: GPUID) -> torch.device:
    if gpu_id >= 0:
        gpu_id = 'cuda:%d' % (gpu_id)
        if torch.cuda.is_available():
            return torch.device(gpu_id)
        else:
            raise GPUNotAvailable(gpu_id)
    else:
        return torch.device('cpu')

def _check_gpu_requirement(flag: bool) -> None:
    if flag:
        if not torch.cuda.is_available():
            raise GPUNotAvailable('cuda')

def make_config(args: ArgsObject) -> BaseConfig:
    project_root = os.path.abspath(os.path.dirname(__file__))
    config = BasicSweeper(args.config_file, project_root).parse(args.id)
    config.device = args.device
    return config

def set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_cuda_deterministic(flag: bool) -> None:
    torch.use_deterministic_algorithms(flag)

def check_device_availability(device: str) -> None:
    if 'cuda' in device and (not torch.cuda.is_available()):
        raise GPUNotAvailable(device)

def initialize_logger(logging_dir: Path, level: int) -> None:

    # create logger with __name__
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # create file handler which logs even debug messages
    handler = logging.FileHandler(logging_dir/LOG_FILE_NAME)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def _try_make_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)

def main(args: List[str]) -> None:
    # print(Path(__file__).parent.absolute())
    
    args = parse_args(args)

    # Config Creation
    project_root = Path(__file__).resolve().parent
    abs_config_file = project_root/args.config_file
    cfg = BasicSweeper(abs_config_file).get_config(args.id)
    #cfg.device = _select_device(args.device)
    cfg.device = _check_gpu_requirement(args.gpu)
    # Logger Initialization
    
    _try_make_dir(cfg.log_dir)
    cfg.recorder = TensorboardRecorder(cfg.log_dir, args.verbose)
    initialize_logger(cfg.log_dir, args.verbose)

    # Log Determination
    set_global_seeds(cfg.seed)
    make_cuda_deterministic(cfg.deterministic)

    algorithm_class = BaseAlgo.retrieve_algo(cfg.algo_class)
    algorithm = algorithm_class(cfg)
    algorithm.run()

if __name__ == '__main__':
    main(sys.argv[1:])