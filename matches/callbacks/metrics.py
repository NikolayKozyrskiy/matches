from collections import defaultdict
from typing import Dict, List, Union
import logging
from warnings import warn

from ignite.distributed import one_rank_only, get_rank
import pandas as pd

from .callback import Callback
from ..utils import dump_json
from ..loop import Loop

LOG = logging.getLogger(__name__)


class BestMetricsReporter(Callback):
    def __init__(
        self, metrics_name_mode: Dict[str, str], log_updates: bool = False
    ) -> None:
        self.metrics_name_mode = metrics_name_mode
        self.log_updates = log_updates
        self.epochs_elapsed_num = 0

        self.metrics_setup = {}
        for name, mode in metrics_name_mode.items():
            self.metrics_setup[name] = {}
            self.metrics_setup[name]["best_epoch_idx"] = None
            if mode == "min":
                self.metrics_setup[name]["best_value"] = float("+inf")
                self.metrics_setup[name]["op"] = min
            else:
                self.metrics_setup[name]["best_value"] = float("-inf")
                self.metrics_setup[name]["op"] = max

    @one_rank_only()
    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        for name, setup in self.metrics_setup.items():
            current_value = loop.metrics.latest[name].value
            better_value = setup["op"](current_value, setup["best_value"])
            if better_value != setup["best_value"]:
                setup["best_value"] = better_value
                setup["best_epoch_idx"] = epoch_no
                if self.log_updates:
                    LOG.info(
                        "Metric %s reached new best value %g, updating checkpoint",
                        name,
                        setup["best_value"],
                    )
        self.epochs_elapsed_num += 1

    @one_rank_only()
    def on_train_end(self, loop: "Loop"):
        summary = self.get_summary()
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(loop.logdir / "best_metrics_summary.csv")
        (loop.logdir / "best_metrics_summary.txt").write_text(str(summary_df))
        dump_json(summary, loop.logdir / "best_metrics_summary.json", indent=2)

    def get_summary(self) -> Dict[str, List[Union[str, float, int]]]:
        if get_rank() != 0:
            warn("get_summary() was called in process with non-zero rank")
        summary = defaultdict(list)
        for name, setup in self.metrics_setup.items():
            summary["metric_name"].append(name)
            summary["best_value"].append(setup["best_value"])
            summary["best_epoch_idx"].append(setup["best_epoch_idx"])
            summary["total_epochs_num"].append(self.epochs_elapsed_num)
        return summary
