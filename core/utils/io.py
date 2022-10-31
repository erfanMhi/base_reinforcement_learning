import logging
from pathlib import Path
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

import yaml

logger = logging.getLogger(__name__)

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.exception('')
        return data

def save_yaml(path: Path, data: Dict[str, Any]) -> bool:
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        return True
    return False