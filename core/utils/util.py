from abc import ABCMeta
from datetime import datetime
from copy import deepcopy
from typing import (
    Generator,
    Tuple,
    Any,
    Dict
)

from core.utils.exceptions import SubclassIDError

def iterate_nested_dict(dic: Dict[Any, Any]) -> Generator[Tuple[Dict[Any, Any], Any, Any], None, None]:
    for param, value in dic.items():
        yield dic, param, value
        if isinstance(value, dict):
            yield from iterate_nested_dict(value)

def retrieve_subclass(cls: ABCMeta, subclass_name: str):

        for subclass in cls.__subclasses__():
            if subclass.__name__ == subclass_name:
                return subclass
        
        raise SubclassIDError(cls, subclass_name)

def get_current_datetime() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def extract_name_and_args(config: dict) -> Tuple[str, dict]:
    """Extracts the name of the class and the arguments from a config dict.
    ----------
    config: dict
        A dictionary containing the name of the class and the arguments.
    Returns
    -------
    name: str
        The name of the class.
    args: dict
        The arguments of the class.
    """
    config = deepcopy(config)
    name = config.pop("name")
    args = config
    return name, args