import argparse
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import numpy as np


TensorType = Union[np.ndarray, torch.Tensor]
INTType = Union[int, np.int32, np.int64]
FLOATType = Union[float, np.float32, np.float64]
ScalarType = Union[int, float]

# Environment data types
ObsType = TensorType
ActionType = Union[TensorType, ScalarType]
RewardType = float
TerminalType = bool
RewardType = ScalarType