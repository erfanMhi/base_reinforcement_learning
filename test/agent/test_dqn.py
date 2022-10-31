import logging
from pathlib import Path
from random import random
from types import SimpleNamespace

import pytest
import shutil
from core.algos import BaseAlgo
from experiments.tuner import BasicTuner


#FIX: Pytest get stuck when I'm running multiprocessing tests on pooling (i.e., increasing the number of workers to 2 or more)
@pytest.mark.parametrize("args", [{
    'config_file': Path('experiments/data/configs/tests/agent/dqn/cartpole.yaml'),
    'gpu': False,
    'verbose': logging.INFO,
    'workers': 1, 
    'run': 1
}])
def test_execution(args): # TODO: Test performance and functionality & improve the convergence rate on cartpole
    args = SimpleNamespace(**args)    
    config_file = args.config_file.absolute()
    tuner = BasicTuner.create_by_config_file(config_file, verbose=args.verbose, workers=args.workers, gpu=args.gpu, run=args.run)
    best_cfg = tuner.tune()
    shutil.rmtree(tuner.log_dir)
    

def test_metrics_storage_capability():
    """Test the major parameters of the algorithm.
    """
    pass

