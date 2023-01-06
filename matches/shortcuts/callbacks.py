from typing import Dict, Optional, Union, Type, Iterable
from tensorboardX import SummaryWriter

from ..callbacks import Callback, TensorboardMetricWriterCallback, BestModelSaver
from ..loop import Loop


def get_summary_writer(loop: Loop) -> SummaryWriter:
    return [
        c.get_sw(loop)
        for c in loop.callbacks
        if isinstance(c, TensorboardMetricWriterCallback)
    ][0]


def get_best_metric(loop: Loop) -> Optional[Dict[str, Union[int, float]]]:
    for c in loop.callbacks:
        if isinstance(c, BestModelSaver):
            return {"value": c.best_value, "epoch": c.best_epoch}
    return None


def has_callback(callbacks: Iterable[Callback], callback_cls: Type[Callback]) -> bool:
    for c in callbacks:
        if isinstance(c, callback_cls):
            return True
    return False
