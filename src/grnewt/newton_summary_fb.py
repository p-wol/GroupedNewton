from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import HgCfg
from .fullbatch import fullbatch_gradient
from .hg import compute_Hg_fullbatch
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure


class NewtonSummaryFB(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        full_loss,
        model,
        final_loss,
        data_loader: DataLoader,
        dataset_size: int,
        *,
        loader_pre_hook,
        cfg: HgCfg,
    ):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y_target) = l(m(x), y_target)
        model: model to train
        final_loss: final loss (NLL, MSE...); must use reduction='mean'
        data_loader: generates the data points used to estimate H and g
        cfg: validated `optimizer.hg` node; see grnewt/config.py.
        """
        self.model = model
        self.final_loss = final_loss
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.full_loss = full_loss
        self.loader_pre_hook = loader_pre_hook
        self.cfg = cfg
        super().__init__(param_groups, {"lr": 0, "damping": cfg.damping})

        self.param_struct = ParamStructure(param_groups)
        self.device = self.param_struct.device
        self.dtype = self.param_struct.dtype

        self.step_counter = 0
        self.logs = {}

    def get_lrs(self):
        return [group["lr"] for group in self.param_groups]

    def damping_mul(self, factor):
        for group in self.param_groups:
            group["damping"] *= factor

    def _init_group(
        self, group: dict[str, Any], params_with_grad: list[Tensor], d_p_list: list[Tensor]
    ):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

    def step(self):
        # Update groups
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            self._init_group(group, params_with_grad, d_p_list)

        # Compute lrs when using the fullbatch gradient direction
        direction = fullbatch_gradient(
            self.param_struct,
            self.final_loss,
            self.model,
            self.data_loader,
            self.dataset_size,
            loader_pre_hook=self.loader_pre_hook,
        )

        lrs = self.compute_lrs(direction)

        # Assign lrs
        for group, lr in zip(self.param_groups, lrs, strict=False):
            lr1 = lr.item()
            if self.cfg.remove_negative:
                lr1 = max(0, lr1)
            group["lr"] = group["damping"] * lr1

        # Perform update
        with torch.no_grad():
            i = 0
            for group in self.param_groups:
                for p in group["params"]:
                    p.add_(direction[i], alpha=-group["lr"])
                    i += 1
        self.step_counter += 1

    def compute_lrs(self, direction):
        H, g, order3 = compute_Hg_fullbatch(
            self.param_struct,
            self.full_loss,
            self.data_loader,
            self.dataset_size,
            direction,
            noregul=self.cfg.noregul,
            loader_pre_hook=self.loader_pre_hook,
        )

        self.logs["H"] = H
        self.logs["g"] = g
        self.logs["order3"] = order3
        self.logs["lrs"] = torch.tensor(
            [group["lr"] for group in self.param_groups], device=self.device, dtype=self.dtype
        )

        # FIX (2026-08-21), three bugs, all on this branch:
        #  1. `nesterov_lrs` returns (lrs, dct_logs); the old code unpacked three
        #     values, so this path raised ValueError before reaching any numerics.
        #  2. it passed the signed `order3` where the diagonal of D, |order3|^(1/3),
        #     is expected -- nesterov.py now rejects that explicitly.
        #  3. it branched on `noregul` alone and ignored `nesterov.use`, so
        #     `nesterov.use=False` still ran the cubic solver, and `ridge` was dead.
        if self.cfg.noregul or not self.cfg.nesterov.use:
            if self.cfg.noregul:
                regul_H = 0
            else:
                regul_H = self.cfg.ridge * torch.eye(
                    H.size(0), dtype=self.dtype, device=self.device
                )
            return torch.linalg.solve(H + regul_H, g)

        nest = self.cfg.nesterov
        lrs, lrs_logs = nesterov_lrs(
            H,
            g,
            order3.abs().pow(1 / 3),
            damping_int=nest.damping_int,
            threshold_D_sing=nest.threshold_D_sing,
            hard_case_rtol=nest.hard_case_rtol,
            refine=nest.refine,
        )
        for k, v in lrs_logs.items():
            self.logs["nesterov." + k] = v
        if not lrs_logs["found"]:
            raise RuntimeError(
                "NewtonSummaryFB: the cubic model is unbounded below (Prop. 2); "
                "there is no fallback lr in the full-batch optimizer."
            )
        return lrs
