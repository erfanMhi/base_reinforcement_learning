
from core.utils.typing import GPUID

class GPUNotAvailable(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, gpu_id: GPUID):
        self.gpu_id = gpu_id
        super().__init__()

    def __str__(self):
        return f'GPU {self.gpu_id} is not available!'


class SubclassIDError(ValueError):
    def __init__(self, cls: object, name: str):
        self.name = name
        self.cls = cls
        super().__init__()

    def __str__(self):
        return f"{self.cls.__name__} doesn't have a sublcass called {self.name}!"