from typing import Callable, Dict, List, Optional, Type, Iterable, Union
from tensorboardX import SummaryWriter

from .metrics import MetricBestSetup
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


def get_metric_best_setups(
    loop: Loop,
) -> Optional[Dict[str, MetricBestSetup]]:
    for c in loop.callbacks:
        if isinstance(c, BestMetricsReporter):
            return c.metric_best_setups_dict
    return None


def get_best_model_metric_setup(loop: Loop) -> Optional[MetricBestSetup]:
    for c in loop.callbacks:
        if isinstance(c, BestModelSaver):
            return c.metric_best_setup
    return None


def has_callback(callbacks: Iterable[Callback], callback_cls: Type[Callback]) -> bool:
    for c in callbacks:
        if isinstance(c, callback_cls):
            return True
    return False


def get_callback(
    callbacks: Iterable[Callback], callback_cls: Type[Callback]
) -> Optional[Callback]:
    for c in callbacks:
        if isinstance(c, callback_cls):
            return c
    return None
