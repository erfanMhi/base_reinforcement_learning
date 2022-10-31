import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import (
    Optional,
    Type
)
from types import TracebackType
from torch import nn

from core.utils.typing import (
    TensorType,
    ScalarType
)

class BaseRecorder(ABC):

    def __init__(self, log_dir: Path, level: int = logging.INFO) -> None:
        self._log_dir = log_dir
        self.level = level

    def add_scalar(self, key: str, scalar: ScalarType, step: int, verbosity_level: int) -> None:
        if self._is_verbose(verbosity_level):
            self._add_scalar(key, scalar, step)

    @abstractmethod
    def _add_scalar(self, key: str, scalar: ScalarType, step: int) -> None: ...

    def add_histogram(self, key: str, values: TensorType, step: int, verbosity_level: int) -> None:
        if self._is_verbose(verbosity_level):
            self._add_histogram(key, values, step)
    
    @abstractmethod
    def _add_histogram(self, key: str, values: TensorType, step: int) -> None: ...

    def add_image(self, key: str, image: TensorType, step: int, verbosity_level: int) -> None:
        if self._is_verbose(verbosity_level):
            self._add_image(key, image, step)
    
    @abstractmethod
    def _add_image(self, key: str, image: TensorType, step: int) -> None: ...
    
    def add_hparams(self, params: dict, metrics: dict, verbosity_level: int) -> None:
        if self._is_verbose(verbosity_level):
            self._add_hparams(params, metrics)
    
    @abstractmethod
    def _add_hparams(self, params: dict, metrics: dict) -> None: ...

    def add_graph(self, model: nn.Module, input_to_model: TensorType, verbosity_level: int) -> None:
        if self._is_verbose(verbosity_level):
            self._add_graph(model, input_to_model)

    @abstractmethod
    def _add_graph(self, model: nn.Module, input_to_model: TensorType = None) -> None: ...

    def _is_verbose(self, verbosity_level: int) -> bool:
        return verbosity_level >= self.level

    @abstractmethod
    def __enter__(self) -> 'BaseRecorder':
        pass
 
    @abstractmethod
    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException], traceback: Optional[TracebackType]) -> bool:
        pass
