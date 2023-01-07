from typing import Dict, List, Optional, Type, Iterable, Union
from tensorboardX import SummaryWriter

from ..callbacks import (
    Callback,
    TensorboardMetricWriterCallback,
    BestMetricsReporter,
    BestModelSaver,
)
from ..loop import Loop


def get_summary_writer(loop: Loop) -> SummaryWriter:
    return [
        c.get_sw(loop)
        for c in loop.callbacks
        if isinstance(c, TensorboardMetricWriterCallback)
    ][0]


def get_metrics_summary(
    loop: Loop,
) -> Optional[Dict[str, List[Union[str, float, int]]]]:
    for c in loop.callbacks:
        if isinstance(c, BestMetricsReporter):
            return c.get_summary()
    return None


def get_best_model_metric(loop: Loop) -> Optional[Dict[str, Union[str, int, float]]]:
    for c in loop.callbacks:
        if isinstance(c, BestModelSaver):
            return {
                "metric_name": c.metric_name,
                "best_value": c.best_value,
                "best_epoch": c.best_epoch,
                "total_epochs": c.epochs_elapsed_num,
            }
    return None


def has_callback(callbacks: Iterable[Callback], callback_cls: Type[Callback]) -> bool:
    for c in callbacks:
        if isinstance(c, callback_cls):
            return True
    return False
