import itertools
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import HgCfg
from .hg import compute_Hg
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure


class NewtonSummary(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        full_loss,
        data_loader: DataLoader,
        updater,
        *,
        loader_pre_hook,
        cfg: HgCfg,
    ):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y) = l(m(x), y)
        data_loader: generates the data points used to estimate H and g
        cfg: validated `optimizer.hg` node; see grnewt/config.py. Every field, its
             default, its meaning and which optimizers read it are declared there.
        """
        self.fn_data_loader = create_infinite_data_loader(data_loader)
        self.dl_iter = iter(self.fn_data_loader())
        self.full_loss = full_loss
        self.updater = updater
        self.loader_pre_hook = loader_pre_hook
        self.cfg = cfg
        self.curr_lrs = 0

        # `damping` is per-group state (damping_mul mutates it), so it belongs to
        # torch's `defaults` mechanism rather than to `self.cfg`.
        super().__init__(param_groups, {"lr": 0, "damping": cfg.damping})

        self.param_struct = ParamStructure(param_groups)
        self.device = self.param_struct.device
        self.dtype = self.param_struct.dtype

        self.step_counter = 0

        if cfg.nesterov.mom_order3_ != 0.0:
            self.order3_ = None

        if cfg.movavg != 0:
            self.H = None
            self.g = None
            self.order3 = None

        self.reset_logs()

    def reset_logs(self):
        if hasattr(self, "logs"):
            del self.logs
        self.logs = {
            "H": [],
            "g": [],
            "order3": [],
            "lrs": [],
            "lrs_clipped": [],
            "curr_lrs": [],
            "nesterov.r": [],
            "nesterov.converged": [],
        }

    def damping_mul(self, factor):
        for group in self.param_groups:
            group["damping"] *= factor
            group["lr"] *= factor

    def step(self):
        direction = self.updater.compute_step()

        # Compute H, g
        perform_update = True
        if self.step_counter % self.cfg.period_hg == 0:
            # Prepare data
            x, y = next(self.dl_iter)
            x, y = self.loader_pre_hook(x, y)

            # Compute H, g, order3
            H, g, order3 = compute_Hg(
                self.param_struct,
                self.full_loss,
                x,
                y,
                direction,
                noregul=self.cfg.noregul,
                diagonal=self.cfg.diagonal,
            )

            # if self.cfg.diagonal:
            #    H = H.diag().diag()

            order3_ = order3.abs().pow(1 / 3)
            if self.cfg.nesterov.mom_order3_ != 0.0:
                if self.order3_ is None:
                    self.order3_ = order3_
                else:
                    r = self.cfg.nesterov.mom_order3_
                    self.order3_ = r * self.order3_ + (1 - r) * order3_
                    order3_ = self.order3_

            if self.cfg.movavg != 0:
                if self.H is None:
                    self.H = H
                    self.g = g
                    self.order3 = order3
                else:
                    r = self.cfg.movavg
                    self.H = r * self.H + (1 - r) * H
                    self.g = r * self.g + (1 - r) * g
                    self.order3 = r * self.order3 + (1 - r) * order3

                    H = self.H
                    g = self.g
                    order3 = self.order3
                order3_ = order3.abs().pow(1 / 3)

            # Compute lrs
            if self.cfg.noregul or not self.cfg.nesterov.use:
                if self.cfg.noregul:
                    regul_H = 0
                else:
                    regul_H = self.cfg.ridge * torch.eye(
                        H.size(0), dtype=self.dtype, device=self.device
                    )
                lrs = torch.linalg.solve(H + regul_H, g)
            else:
                nest = self.cfg.nesterov
                lrs, lrs_logs = nesterov_lrs(
                    H,
                    g,
                    order3_,
                    damping_int=nest.damping_int,
                    threshold_D_sing=nest.threshold_D_sing,
                    hard_case_rtol=nest.hard_case_rtol,
                    refine=nest.refine,
                )

                for k, v in lrs_logs.items():
                    kk = "nesterov." + k
                    if kk not in self.logs.keys():
                        self.logs[kk] = []
                    self.logs[kk].append(v)

                if not lrs_logs["found"]:
                    perform_update = False
                    print("Nesterov did not converge: lr not updated during this step.")
                    # TODO: throw warning?

            if not perform_update:
                lrs = torch.zeros(g.size(0), dtype=self.dtype, device=self.device)

            # To execute even when update_lrs = False? Block #1
            r = self.cfg.mom_lrs if self.step_counter > 0 else 0
            if self.cfg.maintain_true_lrs:
                self.curr_lrs = r * self.curr_lrs + (1 - r) * lrs
                lrs = self.curr_lrs
                if self.cfg.remove_negative:
                    lrs = lrs.relu()
            else:
                if self.cfg.remove_negative:
                    lrs = lrs.relu()
                self.curr_lrs = r * self.curr_lrs + (1 - r) * lrs
                lrs = self.curr_lrs

            # To execute even when update_lrs = False? Block #2
            # Assign lrs
            self.logs["lrs_clipped"].append(lrs)
            self.logs["curr_lrs"].append(self.curr_lrs)
            for group, lr in zip(self.param_groups, lrs):
                group["lr"] = group["damping"] * lr.item()

            # Store logs
            self.logs["H"].append(H)
            self.logs["g"].append(g)
            self.logs["order3"].append(order3)
            self.logs["lrs"].append(
                torch.tensor(
                    [group["lr"] for group in self.param_groups],
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        # Perform update
        with torch.no_grad():
            i = 0
            for group in self.param_groups:
                for p in group["params"]:
                    p.add_(direction[i], alpha=-group["lr"])
                    i += 1

        self.step_counter += 1


def create_infinite_data_loader(data_loader):
    # XXX: if the batch_size does not divide the total number of samples in
    #      the data_loader, then this may fail (possibly batches of irregular sizes)
    def f():
        for dl in itertools.repeat(data_loader):
            yield from dl

    return f


def update_momentum_buffers(
    params: list[Tensor],
    d_p_list: list[Tensor],
    momentum_buffer_list: list[Optional[Tensor]],
    *,
    momentum: float,
    momentum_damp: float,
):
    for i, _param in enumerate(params):
        d_p = d_p_list[i]

        buf = momentum_buffer_list[i]

        if buf is None:
            buf = torch.clone(d_p).detach()
            momentum_buffer_list[i] = buf
        else:
            buf.mul_(momentum).add_(d_p, alpha=1 - momentum_damp)
