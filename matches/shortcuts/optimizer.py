import typing
from enum import Enum
from typing import Optional, Protocol

import torch
from matches.loop import Loop
from torch.optim.optimizer import Optimizer


class SchedulerScopeType(Enum):
    EPOCH = "epoch"
    BATCH = "batch"
    BATCH_AND_EPOCH = "batch_and_epoch"
    NONE = "none"


@typing.runtime_checkable
class LRSchedulerProto(Protocol):
    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def get_last_lr(self):
        pass

    def step(self, epoch=None):
        pass


class LRSchedulerWrapper:
    def __init__(
        self,
        scheduler: Optional[LRSchedulerProto],
        scope_type: SchedulerScopeType,
    ) -> None:
        self._scheduler = scheduler
        self._scope_type = scope_type

    def state_dict(self):
        if self._scheduler is not None:
            return self._scheduler.state_dict()
        else:
            return {}

    def load_state_dict(self, state_dict):
        if self._scheduler is not None:
            self._scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        if self._scheduler is not None:
            return self._scheduler.get_last_lr()

    def step(self, scope_type: SchedulerScopeType, epoch=None):
        if self._scheduler is not None and scope_type == self._scope_type:
            self._scheduler.step(epoch)

    def step_batch(self, epoch=None):
        if self._scheduler is not None and self._scope_type == SchedulerScopeType.BATCH:
            self._scheduler.step(epoch)

    def step_epoch(self, epoch=None):
        if self._scheduler is not None and self._scope_type == SchedulerScopeType.EPOCH:
            self._scheduler.step(epoch)


def simple_gd_step(loop: Loop, optimizer: Optimizer, loss: torch.Tensor):
    loop.backward(loss)
    loop.optimizer_step(optimizer)
