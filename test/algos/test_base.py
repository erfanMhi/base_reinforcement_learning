import logging
from pathlib import Path
from random import random
import shutil
from types import SimpleNamespace
from typing import Tuple

import pytest

from core.algos import BaseAlgo
from core.utils.typing import ScalarType
from experiments.tuner import BasicTuner


class MockAlgo(BaseAlgo):

    def _run(self):
        
        return {'auc': random()}
        
    def _get_metrics_keys(self) -> Tuple[ScalarType]:
        metrics_keys = super()._get_metrics_keys()
        return ['auc'] + metrics_keys



#FIX: Pytest get stuck when I'm running multiprocessing tests on pooling (i.e., increasing the number of workers to 2 or more)
@pytest.mark.parametrize("args", [{
    'config_file': Path('experiments/data/configs/tests/algo/base/mock_algo.yaml'),
    'gpu': False,
    'verbose': logging.INFO,
    'workers': 1, 
    'run': 1
}, {
    'config_file': Path('experiments/data/configs/tests/algo/base/mock_algo.yaml'),
    'gpu': False,
    'verbose': logging.INFO,
    'workers': 1,
    'run': 5
}])
def test_execution(args):
    args = SimpleNamespace(**args)    
    config_file = args.config_file.absolute()
    tuner = BasicTuner.create_by_config_file(config_file, verbose=args.verbose, workers=args.workers, gpu=args.gpu, run=args.run)
    best_cfg = tuner.tune()
    shutil.rmtree(tuner.log_dir)

def test_metrics_storage_capability():
    """Test the major parameters of the algorithm.
    """
    pass

