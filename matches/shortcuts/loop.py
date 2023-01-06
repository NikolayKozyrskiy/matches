from typing import Tuple
from tensorboardX import SummaryWriter

from ..callbacks import TensorboardMetricWriterCallback, BestModelSaver
from ..loop import Loop


def get_summary_writer(loop: Loop) -> SummaryWriter:
    return [
        c.get_sw(loop)
        for c in loop.callbacks
        if isinstance(c, TensorboardMetricWriterCallback)
    ][0]


def get_best_metric(loop: Loop) -> Tuple[float, int]:
    for c in loop.callbacks:
        if isinstance(c, BestModelSaver):
            return c.best_value, c.best_epoch
