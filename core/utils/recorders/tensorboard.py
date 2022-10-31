import logging
from pathlib import Path
from types import TracebackType
from typing import (
    Optional,
    Tuple,
    Type
)

import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tbparse import SummaryReader

from core.utils.recorders import BaseRecorder
from core.utils.typing import (
    TensorType,
    ScalarType
)

logger = logging.getLogger(__name__)

class TensorboardRecorder(BaseRecorder):

    def __init__(self, log_dir: Path, level: int = logging.INFO) -> None:
        super(TensorboardRecorder, self).__init__(log_dir, level)
        self._writer = SummaryWriter(self._log_dir)

    def _add_scalar(self, key: str, scalar: ScalarType, step: int) -> None:
        self._writer.add_scalar(key, scalar, step)

    def _add_histogram(self, key: str, values: TensorType, step: int) -> None:
        self._writer.add_histogram(key, values, step)

    def _add_image(self, key: str, image: TensorType, step: int) -> None:
        self._writer.add_image(key, image, step)

    def _add_hparams(self, params: dict, metrics: dict) -> None:
        self._writer.add_hparams(params, metrics)

    def _add_graph(self, model: nn.Module, input_to_model: TensorType) -> None:
        self._writer.add_graph(model, input_to_model)

    @staticmethod
    def retrieve_scalar(log_dir: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
        scalars = SummaryReader(log_dir).scalars
        key_mask = scalars['tag'] == key
        scalars = scalars[key_mask]
        scalars = scalars.drop(columns=['tag'])
        steps = scalars['step'].to_numpy()
        values = scalars['value'].to_numpy()
        return steps, values

    @staticmethod
    def retrieve_metric_value(log_dir: str, key: str) -> float:
        _, values = TensorboardRecorder.retrieve_scalar(log_dir, key)
        return values[0]

    def __enter__(self) -> BaseRecorder:
        if self._writer is None:
            self._writer = SummaryWriter(self.log_dir)
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException], traceback: Optional[TracebackType]) -> bool:
        self.writer.flush()
        self.writer.close()
        if exception_type is not None:
            logger.error(traceback)
            return False
        return True