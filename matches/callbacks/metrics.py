import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union
from warnings import warn

import pandas as pd
from ignite.distributed import get_rank, one_rank_only

from ..loop import Loop
from ..shortcuts.metrics import MetricBestSetup
from ..utils import dump_json
from .callback import Callback

LOG = logging.getLogger(__name__)


class BestMetricsReporter(Callback):
    def __init__(
        self,
        metrics_name_mode: Dict[str, str],
        log_updates: bool = False,
        summary_formats: List[str] = ("txt", "csv"),
    ) -> None:
        self.metrics_name_mode = metrics_name_mode
        self.log_updates = log_updates
        self.summary_formats = summary_formats
        self.epochs_elapsed_num = 0

        self.metric_best_setups_dict = {
            name: MetricBestSetup(name, mode)
            for name, mode in metrics_name_mode.items()
        }

    @one_rank_only()
    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        for name, setup in self.metric_best_setups_dict.items():
            if (
                setup.update(loop.metrics.latest[name].value, epoch_no)
                and self.log_updates
            ):
                LOG.info(
                    "Metric %s reached new best value %g at epoch %d",
                    name,
                    setup.best_value,
                    epoch_no,
                )
        self.epochs_elapsed_num += 1

    @one_rank_only()
    def on_train_end(self, loop: "Loop"):
        summary = self.get_summary()
        summary_df = pd.DataFrame(summary)
        if "csv" in self.summary_formats:
            summary_df.to_csv(loop.logdir / "best_metrics_summary.csv")
        if "txt" in self.summary_formats:
            (loop.logdir / "best_metrics_summary.txt").write_text(str(summary_df))
        # dump_json(summary, loop.logdir / "best_metrics_summary.json", indent=2)

    def get_summary(self) -> Dict[str, List[Union[str, float, int]]]:
        if get_rank() != 0:
            warn("get_summary() was called in process with non-zero rank")
        summary = defaultdict(list)
        for name, setup in self.metric_best_setups_dict.items():
            summary["metric_name"].append(name)
            summary["best_value"].append(setup.best_value)
            summary["best_epoch_idx"].append(setup.best_epoch_idx)
            summary["total_epochs_num"].append(setup.total_epochs_num)

        return summary
