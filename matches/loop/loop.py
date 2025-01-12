import logging
import os
import typing
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)
from warnings import warn

import ignite.distributed as idist
import torch
from ignite.metrics import Metric
from ignite.utils import convert_tensor
from matches.accelerators import Accelerator
from matches.loop.iteration import IterationCounter
from matches.loop.loader_scheduling import DataloaderOverrider
from matches.loop.metric_manager import MetricManager
from matches.shortcuts.module import module_eval, module_train
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from matches.callbacks.callback import Callback

LOG = logging.getLogger(__name__)

T_batch = TypeVar("T_batch")


@typing.runtime_checkable
class StateSource(Protocol):
    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass


NON_BLOCKING_COPY = bool(os.environ.get("NBC", "True"))


class StateManager:
    def __init__(self):
        self._state_sources: Dict[str, StateSource] = {}

    def attach(self, key: str, source: StateSource):
        self._state_sources[key] = source

    def state_dict(self, skip_keys: Optional[Sequence[str]] = None):
        skip_keys = skip_keys if skip_keys else []
        return {
            key: value.state_dict()
            for key, value in self._state_sources.items()
            if key not in skip_keys
        }

    def state_dict_by_key(self, key: str):
        return {key: self._state_sources[key].state_dict()}

    def load_state_dict(self, state_dict, skip_keys: Optional[Sequence[str]] = None):
        skip_keys = skip_keys if skip_keys else []
        for k, ss in self._state_sources.items():
            if k not in skip_keys:
                ss.load_state_dict(state_dict[k])

    def load_state_dict_by_key(self, state_dict, key: str):
        self._state_sources[key].load_state_dict(state_dict[key])

    def write_state(
        self, file: Union[str, PathLike], skip_keys: Optional[Sequence[str]] = None
    ):
        with Path(file).open("wb") as f:
            state_dict = self.state_dict(skip_keys=skip_keys)
            if len(state_dict) == 0:
                warn(
                    "state_dict is empty. Did you forget to attach "
                    "model/optimizer/scheduler etc?"
                )
            torch.save(state_dict, f)

    def write_state_by_key(self, file: Union[str, PathLike], key: str):
        with Path(file).open("wb") as f:
            state_dict = self.state_dict_by_key(key=key)
            if len(state_dict) == 0:
                warn(
                    "state_dict is empty. Did you forget to attach "
                    "model/optimizer/scheduler etc?"
                )
            torch.save(state_dict, f)

    def read_state(
        self, file: Union[str, PathLike], skip_keys: Optional[Sequence[str]] = None
    ):
        with Path(file).open("rb") as f:
            self.load_state_dict(torch.load(f, map_location="cpu"), skip_keys)

    def read_state_by_key(self, file: Union[str, PathLike], key: str):
        with Path(file).open("rb") as f:
            self.load_state_dict_by_key(torch.load(f, map_location="cpu"), key)


class Loop:
    def __init__(
        self,
        logdir: PathLike,
        callbacks: List["Callback"],
        loader_override: str = "disabled",
        attach_iterations: bool = True,
    ):
        self.callbacks = callbacks
        self.state_manager = StateManager()
        self.metrics = MetricManager(self)
        """Object for metrics logging"""

        self.logdir: Path = Path(logdir)

        self.iterations = IterationCounter()
        if attach_iterations:
            self.attach(iterations=self.iterations)

        self._in_epoch = False
        self._in_dataloader = False
        self._mode: Optional[str] = None

        self._modules: List[nn.Module] = []
        self._loader_override = DataloaderOverrider(loader_override)

    def _emit_event(self, event: str, **event_kwargs):
        for c in self.callbacks:
            getattr(c, event)(self, **event_kwargs)

    def attach(
        self,
        id: str = None,
        obj: Any = None,
        objects_dict: typing.Mapping[str, Any] = None,
        **kwargs,
    ):
        """Add object to list of managed objects.

        Object with `state_dict()/load_state_dict()` will have their state saved and restored.
        For `torch.nn.Module` objects their `train()/eval()` mode will be managed.

        Examples::

            loop.attach("model", model)
            loop.attach(optimizer=optimizer)
            loop.attach(objects_dict={"model/generator": generator}

        Args:
            id: unique identifier for object
            obj: Object itself
            objects_dict: dict where keys are ids and values are objects
            **kwargs: arg-name -- id, arg-value -- object

        Returns:
              None

        """
        objects_dict = objects_dict or {}
        if id and obj is not None:
            if isinstance(obj, nn.Module):
                self._modules.append(obj)
            if isinstance(obj, StateSource) and not isinstance(obj, Metric):
                self.state_manager.attach(id, obj)

        kwargs.update(objects_dict)
        for k, v in kwargs.items():
            self.attach(k, v)

    def iterate_epochs(self, epochs) -> Iterable[int]:
        """Iterate over epochs

        * Emits callback events `on_epoch_start/end`

        Yields:
              current epoch number
        """
        if self._loader_override.mode == "short":
            epochs = 2

        while self.iterations.current_epoch < epochs:
            self.metrics.reset()
            self._in_epoch = True
            with self._wrap_in_events(
                "on_epoch_start",
                "on_epoch_end",
                epoch_no=self.iterations.current_epoch,
                total_epochs=epochs,
            ):
                yield int(self.iterations.current_epoch)
            self.iterations.current_epoch.inc()
            self._in_epoch = False

    def iterate_dataloader(
        self, dataloader: DataLoader[T_batch], mode="valid", move_to_default_device=True
    ) -> Iterable[T_batch]:
        """Iterate dataloader batches.

        Depending on chosen mode following is true inside loop over this generator:

        * Emits callback events `on_iteration_start/end`
        * Moves all torch.Tensor objects in batch to default device
        * Handles set_grad_enabled an train/eval of attached :obj:`nn.Module`:

            * mode="train" trigger train() on all `torch.nn.Modules` attached to
              `Loop`. Gradients are also enabled.
            * mode="valid" trigger eval() on all `torch.nn.Modules` attached to
              `Loop`. Gradients are also disabled.


        Args:
            dataloader: dataloader to iterate
            mode: "train" or "valid"
            move_to_default_device: Controls whether all tensor in batches are automatically moved
                to current device. Default `True`.

        Yields:
            Batches from dataloader.
        """
        self._mode = mode

        assert (
            mode != "train" or self._in_epoch
        ), "Dataloader can be run in train mode only inside epoch loop"

        # dataloader = self._loader_override(dataloader, mode)

        with self._wrap_in_events(
            "on_dataloader_start", "on_dataloader_end", dataloader=dataloader
        ), self.mode(mode):
            self._in_dataloader = True
            for batch_no, batch in enumerate(dataloader):
                with self._wrap_in_events(
                    "on_iteration_start", "on_iteration_end", batch_no=batch_no
                ):
                    if move_to_default_device:
                        batch = convert_tensor(
                            batch, idist.device(), non_blocking=NON_BLOCKING_COPY
                        )
                    yield batch
                    if self._mode == "train":
                        self.iterations.current_batch.inc()
            self._in_dataloader = False

    @contextmanager
    def _wrap_in_events(self, enter_event_name, exit_event_name, **kwargs):
        """CM for emits enter_event_name before block, and emits exit_event_name
        after successful generator termination
        (even if it was stopped with continue statement)
        """
        try:
            self._emit_event(enter_event_name, **kwargs)
            yield
            self._emit_event(exit_event_name, **kwargs)
        except GeneratorExit:
            self._emit_event(exit_event_name, **kwargs)
            raise

    def backward(self, loss: torch.Tensor, **backward_kwargs):
        """Simple optional wrapper for backward call.

        Calls backward and emits event. Most likely will have more logic down the road.
        So it's recommended to use it even now

        Args:
            loss: loss to call backward on
            **backward_kwargs:

        Returns:
              None
        """
        loss.backward(**backward_kwargs)
        self._emit_event("on_after_backward")

    @contextmanager
    def mode(self, mode="valid"):
        """
        Contextmanager managing gradients and train/eval mode of attached modules

        Args:
            mode: train/valid
        """
        old_mode = self._mode
        is_eval = mode == "valid"
        try:
            self._mode = mode
            if is_eval:
                with torch.set_grad_enabled(False), module_train(
                    *self._modules, train=False
                ):
                    yield
            else:
                yield
        except GeneratorExit:
            pass

        self._mode = old_mode

    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Optional[Callable[[], float]] = None,
        zero_grad: Union[bool, str] = True,
    ):
        """Optional shortcut wrapper for optimizer step.

        * Zeroes grad after step (quite common case, you know ;-))
        * Emits some callback events (before/after step)
        * Counts how much global_steps are done (differs from iterations/batches
          if you implement eg grad accumulation). This counter can be used as
          MetricsIterationType

        Args:
            optimizer: Optimizer to perform step
            closure: Some optimizers require closure
            zero_grad: Set False if you don't want to zero grad for some reason. True by default

        Returns:
              None
        """
        self._emit_event("on_before_optimizer_step", optimizer=optimizer)
        optimizer.step(closure)
        if zero_grad:
            optimizer.zero_grad(zero_grad == "set_to_none")
        self._emit_event("on_after_optimizer_step", optimizer=optimizer)
        self.iterations.global_steps.inc()

    def zero_grad_backward_step(
        self,
        loss: torch.Tensor,
        optimizer: Optimizer,
        set_to_none: bool = True,
        closure: Optional[Callable[[], float]] = None,
        **backward_kwargs,
    ) -> None:
        """Optional shortcut wrapper for optimizer step.

        * Zeroes grad at the beginning of routine (quite common case, you know ;-))
        * Emits some callback events (before/after backward and step)
        * Counts how much global_steps are done (differs from iterations/batches
          if you implement eg grad accumulation). This counter can be used as
          MetricsIterationType

        Args:
            loss: loss to call backward on
            set_to_none: flag passed to optimizer.zero_grad(set_to_none=set_to_none)
            optimizer: Optimizer to perform step
            closure: Some optimizers require closure
             **backward_kwargs:

        Returns:
              None
        """
        optimizer.zero_grad(set_to_none=set_to_none)

        loss.backward(**backward_kwargs)
        self._emit_event("on_after_backward")

        self._emit_event("on_before_optimizer_step", optimizer=optimizer)
        optimizer.step(closure)
        self._emit_event("on_after_optimizer_step", optimizer=optimizer)

        self.iterations.global_steps.inc()

        return None

    def run(self, training_procedure: typing.Callable, *args, **kwargs):
        self._emit_event("on_train_start")
        training_procedure(self, *args, **kwargs)
        self._emit_event("on_train_end")

    def launch(self, program: typing.Callable, accelerator: Accelerator, **kwargs):
        """Launch training program on chosen accelerator

        Examples::

            def train(loop: Loop):
                ...

            loop.launch(train, DDPAccelerator("gpu-id_1,gpu-id_2")

        Args:
              program: Training program
              accelerator: object defining accelerator environment
              **kwargs: additional program kwargs

        """
        self.logdir.mkdir(exist_ok=True, parents=True)
        accelerator.execute(program, loop=self, **kwargs)
