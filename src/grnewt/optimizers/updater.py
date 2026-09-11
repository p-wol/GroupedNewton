import torch
from torch import Tensor
from torch.optim import Optimizer


class Updater(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
                
    def compute_step(self, closure = None):
        raise NotImplementedError

    def step(self, tup_updates):
        """Apply an update returned by `compute_step`.

        `compute_step` returns a descent direction u >= 0 aligned with the
        gradient; the caller is responsible for the minus sign.
        """

        with torch.no_grad():
            j = 0
            for group in self.param_groups:
                for _i, param in enumerate(group["params"]):
                    param.sub_(tup_updates[j])
                    j += 1
