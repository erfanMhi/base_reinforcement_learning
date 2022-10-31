import multiprocessing
import os
import time
from typing import Any, Dict

import torch
import pytest

def print_dict(d: Dict[str, Any]) -> None:
    time.sleep(d['wait_time'])
    return d

def d_generator() -> Dict[str, Any]:
    d1 = {'a': {'c': 3}, 'b': 2, 'wait_time': 5}
    d2 = {'c': 3, 'd': 4, 'wait_time': 1}
    d3 = {'e': 5, 'f': 6, 'wait_time': 10}
    yield d1
    yield d2
    yield d3

@pytest.mark.skip(reason="This test is not part of the test suite: it is used for understanding how pytest interacts with multiprocessing")
def test_pool_workers():
    workers_num = max(os.cpu_count(), torch.cuda.device_count())
    workers_num = 2
    generator = d_generator()
    with multiprocessing.Pool(workers_num) as p:
        results = p.map(print_dict, generator)

    results = list(results)