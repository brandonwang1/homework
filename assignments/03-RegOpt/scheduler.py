from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler.
    """

    def __init__(
        self, optimizer, last_epoch=-1, step_size=2500, max_lr=0.002, min_lr=0.0005
    ):
        """
        Create a new scheduler for cyclic learning rate.

        Arguments:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        last_epoch (int): The index of the last epoch.
        step_size (int): The number of iterations within half a cycle.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.

        """
        self.step_size = step_size
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_counter = 0
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get the current learning rate.
        """
        self.cycle_counter += 1
        if self.cycle_counter > 2 * self.step_size:
            self.cycle_counter = 0

        if self.cycle_counter <= self.step_size:
            # Increasing learning rate within the cycle
            lr = (
                self.min_lr
                + self.cycle_counter * (self.max_lr - self.min_lr) / self.step_size
            )
        else:
            # Decreasing learning rate within the cycle
            lr = (
                self.max_lr
                - (self.cycle_counter - self.step_size)
                * (self.max_lr - self.min_lr)
                / self.step_size
            )

        return [lr for _ in self.base_lrs]
