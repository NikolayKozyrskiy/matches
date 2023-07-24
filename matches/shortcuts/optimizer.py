import typing
from enum import Enum
from typing import Protocol

import torch
from matches.loop import Loop
from torch.optim.optimizer import Optimizer


class SchedulerScopeType(Enum):
    EPOCH = "epoch"
    BATCH = "batch"
    BATCH_AND_EPOCH = "batch_and_epoch"


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
        self, scheduler: LRSchedulerProto, scope_type: SchedulerScopeType
    ) -> None:
        self._scheduler = scheduler
        self._scope_type = scope_type

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        return self._scheduler.get_last_lr()

    def step(self, scope_type: SchedulerScopeType, epoch=None):
        if scope_type == self._scope_type:
            self._scheduler.step(epoch)


def simple_gd_step(loop: Loop, optimizer: Optimizer, loss: torch.Tensor):
    loop.backward(loss)
    loop.optimizer_step(optimizer)
