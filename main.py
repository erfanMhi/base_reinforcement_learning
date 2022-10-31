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
from core.utils.typing import ArgsObject
from core.utils.recorders import TensorboardRecorder
from experiments.tuner.basic import BasicTuner

logger = logging.getLogger(__name__)

def _parse_args(args: List[Any]) -> ArgsObject:

    parser = argparse.ArgumentParser(description="The main file for running experiments")

    parser.add_argument('--config-file', dest='config_file', type=Path, required=True, help='Expect a json file describing the fixed and sweeping parameters')

    parser.add_argument('--gpu', action='store_true', help="Use GPU: if not specified, use CPU")

    parser.add_argument('--verbose', dest='verbose', type=_to_logging_level, required=True, help="Logging level: info or debug")

    parser.add_argument('--workers', dest='workers', default=-1, type=int, help="Number of workers used to run the experiments. -1 means that the number of runs are going to be automatically determined")

    parser.add_argument('--run', dest='run', default=1, type=int, help="Number of times that each algorithm needs to be evaluated")

    args = parser.parse_args(args)

    return args

def _to_logging_level(level: str) -> int:
    if level == 'info':
        return logging.INFO
    elif level == 'debug':
        return logging.DEBUG
    else:
        raise ValueError("Unknown logging level: {}".format(level))
    

def _check_device_availability(flag: bool) -> None:
    if flag and not torch.cuda.is_available():
        raise ValueError("GPU is not available!")

def main(args: List[str]) -> None:

    args = _parse_args(args)

    _check_device_availability(args.gpu)
    
    config_file = args.config_file.absolute()
    tuner = BasicTuner.create_by_config_file(config_file, verbose=args.verbose, workers=args.workers, gpu=args.gpu, run=args.run)
    tuner.tune()

if __name__ == '__main__':
    main(sys.argv[1:])