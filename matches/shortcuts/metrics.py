from typing import Callable, Dict, Union, Optional


class MetricBestSetup:
    def __init__(
        self,
        name: str,
        mode: str,
    ) -> None:
        self.name = name
        self.mode = mode
        self.best_epoch_idx = None
        self.total_epochs_num = 0
        if mode == "min":
            self.compare_op = min
            self.best_value = float("+inf")
        else:
            self.compare_op = max
            self.best_value = float("-inf")

    def update(self, new_value: float, epoch_idx: int) -> bool:
        self.total_epochs_num += 1
        better_value = self.compare_op(new_value, self.best_value)
        if better_value != self.best_value:
            self.best_value = better_value
            self.best_epoch_idx = epoch_idx
            return True
        return False

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "metric_name": self.name,
            "best_value": self.best_value,
            "best_epoch_idx": self.best_epoch_idx,
            "total_epochs_num": self.total_epochs_num,
        }
