from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb
from ignite.distributed import one_rank_only
from wandb.sdk.wandb_run import Run

from ..loop import Loop
from ..loop.metric_manager import MetricEntry
from . import Callback


class WandBLoggingSink(Callback):
    def __init__(
        self,
        comment: str,
        config_dict: Dict[str, Any],
        discard_debug_internal: bool = True,
        log_freq: Optional[int] = 100,
    ):
        self.config_dict = config_dict
        self.comment = comment
        self.discard_debug_internal = discard_debug_internal
        self.log_freq = log_freq
        self.model_watched = False
        self.run = None
        self.loader_mode = None

    @one_rank_only()
    def on_train_start(self, loop: "Loop"):
        self.loader_mode = loop._loader_override.mode
        if self.loader_mode == "disabled":
            self.loader_mode = "full"

        self.config_dict["logdir"] = str(loop.logdir)
        settings = {"silent": True, "console": "off"}
        if self.discard_debug_internal:
            log_internal = (loop.logdir / "_dev_null").resolve()
            log_internal.symlink_to(Path("/dev/null").resolve())
            settings["log_internal"] = str(log_internal)
        self.run: Run = wandb.init(
            dir=str(loop.logdir),
            name=self.comment,
            tags=[self.loader_mode],
            settings=wandb.Settings(**settings),
            config=self.config_dict,
        )

    @one_rank_only()
    def on_train_end(self, loop: "Loop"):
        self._consume_new_entries(loop)
        self.run.finish()
        print("WandBLoggingSink callback has finished!")

    @one_rank_only()
    def on_iteration_end(self, loop: "Loop", batch_no: int):
        self._consume_new_entries(loop)

    @one_rank_only()
    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        self._consume_new_entries(loop)

    @one_rank_only()
    def on_epoch_start(self, loop: "Loop", epoch_no: int, total_epochs: int):
        if not self.model_watched and self.log_freq > 0:
            self.model_watched = True
            wandb.watch(
                loop._modules,
                log="all",
                log_freq=10 if self.loader_mode != "full" else self.log_freq,
            )

    def _consume_new_entries(self, loop: "Loop", reset: bool = True):
        entries: List[MetricEntry] = loop.metrics.collect_new_entries(reset=reset)
        for e in entries:
            entry = {
                e.name: e.value,
                **{t.value: v for t, v in e.iteration_values.items()},
            }
            self.run.log(entry, commit=False)
        self.run.log({}, commit=True)
