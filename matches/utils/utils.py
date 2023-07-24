import json
import os
import random
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


def unique_logdir(root: Union[PathLike, str], comment: str = "") -> Path:
    """Get unique logdir under root based on comment and timestamp"""
    now = datetime.now().strftime("%y%m%d_%H%M")

    return Path(root) / comment / now


def seed_everything(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def setup_cudnn_reproducibility(
    deterministic: bool = None, benchmark: bool = None
) -> None:
    """
    Prepares CuDNN benchmark and sets CuDNN
    to be deterministic/non-deterministic mode
    See https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking

    Borrowed https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/torch.py#L256

    Args:
        deterministic: deterministic mode if running in CuDNN backend.
        benchmark: If ``True`` use CuDNN heuristics to figure out
            which algorithm will be most performant
            for your model architecture and input.
            Setting it to ``False`` may slow down your training.
    """
    if torch.cuda.is_available():
        if deterministic is None:
            deterministic = os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
        torch.backends.cudnn.deterministic = deterministic

        if benchmark is None:
            benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        torch.backends.cudnn.benchmark = benchmark


def makedir(dir_path: Union[str, Path]) -> None:
    dir_path = Path(dir_path).resolve()
    dir_path.mkdir(exist_ok=True, parents=True)


def dump_json(
    data: Dict[str, Any], dst_path: Union[str, Path], indent: int = 2
) -> None:
    dst_path = dst_path if str(dst_path).endswith(".json") else f"{dst_path}.json"
    with open(dst_path, "w") as f:
        json.dump(data, f, indent=indent)
