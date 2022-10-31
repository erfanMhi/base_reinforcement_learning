import shutil
from typing import (
    List,
    Dict,
    Any
)

from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

from experiments.tuner import BasicTuner
from core.utils.configs import DQNConfig
from core.utils.exceptions import SubclassIDError
from core.utils.io import load_yaml, save_yaml
from test.algos import MockAlgo

BASE_DQN_CONFIG = {
    'algo_class': 'MockAlgo',
    'max_steps': 100000,
    'id': 0,
    'param_setting': 0,
    'seed': 0,
    'exploration': {
        'type': 'epislon-greedy',
        'epsilon': 0.1
    },
    'optimizer': {
        'type': 'adam',
        'lr': 0.0001,
        'momentum': 0.9
    },
    'target_net': {
        'type': 'discrete',
        'update_frequency': 8
}}

@pytest.fixture
def test_ids() -> List[int]:
    return [id for id in range(100)]

def _get_config(yaml_file: Path, id: int):
    basic_sweeper = BasicTuner.create_by_config_file(yaml_file)
    return basic_sweeper._get_config(id)

#TODO: Handle undefined search keys
@pytest.mark.parametrize("yaml_dir",[
                        'experiments/data/configs/tests/sweeper/basic/incorrect/algo_params',
                        'experiments/data/configs/tests/sweeper/basic/incorrect/meta_params',
                        'experiments/data/configs/tests/sweeper/basic/incorrect/first_level_keys',
                        'experiments/data/configs/tests/sweeper/basic/incorrect/config_param'
                        ])
def test_incorrect_yaml_files(yaml_dir: str, test_ids: List[int]) -> None:
    yaml_dir = Path(yaml_dir)
    all_raised_exception = True
    for yaml_file in yaml_dir.glob('*.yaml'):
        try:
            for config_id in test_ids:
                config = _get_config(yaml_file, config_id)
            all_raised_exception = False
        except (TypeError, SubclassIDError, KeyError) as err:
            print(f"f{yaml_file} encountered this error: {err}")


    assert all_raised_exception

@pytest.mark.parametrize("yaml_dir",[
                        'experiments/data/configs/tests/sweeper/basic/correct',
                        ])
def test_correct_yaml_files(yaml_dir: str, test_ids: List[int]) -> None:
    yaml_dir = Path(yaml_dir)
    zero_raised_exception = True
    for yaml_file in yaml_dir.glob('*.yaml'):
        try:
            for config_id in test_ids:
                config = _get_config(yaml_file, config_id)
        except (TypeError, SubclassIDError, KeyError) as err:
            print(f"f{yaml_file} encountered this error: {err}")
            zero_raised_exception = False
        finally:
            shutil.rmtree(config.root_log_dir)
    assert zero_raised_exception

@pytest.mark.parametrize("yaml_file, params",[
                        (
                            'experiments/data/configs/tests/sweeper/basic/correct/dqn.yaml', {
                            'id': 0,
                            'param_setting': 0,
                            'seed': 0,
                            'optimizer': {
                                'type': 'adam',
                                'lr': 0.0001,
                                'momentum': 0.9
                            },
                            'target_net': {
                                'type': 'discrete',
                                'update_frequency': 8
                            }}
                        ), (
                            'experiments/data/configs/tests/sweeper/basic/correct/dqn.yaml', {
                            'id': 1,
                            'param_setting': 1,
                            'seed': 0,
                            'optimizer': {
                                'type': 'adam',
                                'lr': 0.001,
                                'momentum': 0.9
                            },
                            'target_net': {
                                'type': 'discrete',
                                'update_frequency': 8
                            }}
                        ), (
                            'experiments/data/configs/tests/sweeper/basic/correct/dqn.yaml', {
                            'id': 2,
                            'param_setting': 2,
                            'seed': 0,
                            'optimizer': {
                                'type': 'adam',
                                'lr': 0.0001,
                                'momentum': 0.9
                            },
                            'target_net': {
                                'type': 'discrete',
                                'update_frequency': 16
                            }}
                        ), (
                            'experiments/data/configs/tests/sweeper/basic/correct/dqn.yaml', {
                            'id': 4,
                            'param_setting': 0,
                            'seed': 1,
                            'optimizer': {
                                'type': 'adam',
                                'lr': 0.0001,
                                'momentum': 0.9
                            },
                            'target_net': {
                                'type': 'discrete',
                                'update_frequency': 8
                            }}
                        )])
def test_parameters_correctness(yaml_file: str, params: Dict[str, Any]) -> None:

    if yaml_file.endswith('dqn.yaml'):
        params = dict(BASE_DQN_CONFIG, **params)
    else:
        raise ValueError(f"yaml_file {yaml_file} not supported")

    yaml_file = Path(yaml_file) 

    id = params['id']
    for key, value in params.items():  
        config = _get_config(yaml_file, id)
        shutil.rmtree(config.root_log_dir)
        assert value == getattr(config, key)
    