from typing import Optional
from warnings import warn

from ignite.distributed import get_rank, one_rank_only
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from ..loop import Loop
from . import Callback


class TensorboardMetricWriterCallback(Callback):
    def __init__(self, logdir_suffix: str = ""):
        self.logdir_suffix = logdir_suffix

        self.sw: Optional[SummaryWriter] = None

    @one_rank_only()
    def on_iteration_end(self, loop: "Loop", batch_no: int):
        self._consume_new_entries(loop)

    @one_rank_only()
    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        self._consume_new_entries(loop)
        self.sw.flush()

    @one_rank_only()
    def on_dataloader_end(self, loop: "Loop", dataloader: DataLoader):
        self._consume_new_entries(loop)
        self.sw.flush()

    @one_rank_only()
    def on_train_end(self, loop: "Loop"):
        if self.sw:
            self.sw.close()
            self.sw = None

    def get_sw(self, loop: "Loop") -> SummaryWriter:
        if get_rank() != 0:
            warn("SummaryWriter was requested used in process with non-zero rank")
        self._init_sw(loop)
        return self.sw

    def _init_sw(self, loop: Loop) -> None:
        if self.sw is None:
            path = loop.logdir / self.logdir_suffix
            path.mkdir(exist_ok=True, parents=True)
            self.sw = SummaryWriter(str(path))

    def _consume_new_entries(self, loop: "Loop") -> None:
        self._init_sw(loop)
        entries = loop.metrics.collect_new_entries()
        for e in entries:
            self.sw.add_scalar(e.name, e.value, global_step=e.iteration)
