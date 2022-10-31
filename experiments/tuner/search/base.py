
from typing import (
    List,
    Any,
    Union
)
from abc import ABC, abstractmethod

import numpy as np

class BaseSearch(ABC):

    @abstractmethod
    def get_search_params(self, random_generator: np.random.Generator) -> List[Any]:
        pass

class GridSearch(BaseSearch):

    def __init__(self, params_list: List[Any]) -> None:
        self.params_list = params_list

    def get_search_params(self, random_generator: np.random.Generator) -> List[Any]:
        return list(self.params_list)

class UniformSearch(BaseSearch):

    def __init__(self, params_list: List[Union[int, float]]) -> None:
        self.params_list = params_list
        self._validate_params_list()
        self.low: float = self.params_list[0]
        self.high: float = self.params_list[1]
        self.size: int = self.params_list[2]

    def _validate_params_list(self):
        valid_params_num = len(self.params_list) == 3
        if not valid_params_num:
            TypeError(f'{UniformSearch.__name__} expected a list with 3 values, got {len(self.params_list)}')

    def get_search_params(self, random_generator: np.random.Generator) -> List[Any]:
        return list(random_generator.uniform(self.low, self.high, self.size))
        