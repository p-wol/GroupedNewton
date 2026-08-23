import itertools
import dataclasses
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .config import HgCfg
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure


@dataclass(kw_only=True, slots=True)
class UpdateInstructions:
    recompute_lrs: bool
    do_update: bool

def increment_step(func):
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.step_counter += 1
        return ret

    return wrapper


class NSBase(torch.optim.Optimizer):
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
        cfg: validated `optimizer.hg` node; see grnewt/config.py for every field,
             its default, and which optimizers read it. Nothing else is read from
             the config, and every field this optimizer ignores is rejected at
             composition time by grnewt.config.check_consumed.
        """
        self.fn_data_loader = create_infinite_data_loader(data_loader)
        self.dl_iter = iter(self.fn_data_loader())
        self.full_loss = full_loss
        self.updater = updater
        self.loader_pre_hook = loader_pre_hook
        self.cfg = cfg
        self.curr_lrs = 0

        # `damping` is per-parameter-group state, not a hyperparameter of the run:
        # damping_mul() mutates it. It therefore belongs to torch's `defaults`
        # mechanism, not to `self.cfg`. Every other field stays in `self.cfg`.
        super().__init__(param_groups, {"lr": 0, "damping": cfg.damping})

        self.param_struct = ParamStructure(param_groups)
        self.device = self.param_struct.device
        self.dtype = self.param_struct.dtype

        # See ParamStructure.build_reindex.
        self._dir_perm = self.param_struct.build_reindex(
            [p for gr in updater.param_groups for p in gr["params"]]
        )

        self.step_counter = 0

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

    def compute_avg_Hg(self, direction):
        raise NotImplementedError

    @increment_step
    def step(self):
        # Compute the direction
        direction = self.param_struct.reindex(self.updater.compute_step(), self._dir_perm)

        # Compute the averages of H, g, order3
        H, g, order3, update_instr = self.compute_avg_Hg(direction)

        # Store logs of H, g, order3
        # XXX: warning: the number of elements in the list may be unpredictible, to FIX
        if H is not None: self.logs["H"].append(H)
        if g is not None: self.logs["g"].append(g)
        if order3 is not None: self.logs["order3"].append(order3)

        # Compute lrs if self.compute_avg_Hg says so
        if update_instr.recompute_lrs:
            # Compute order3_
            order3_ = order3.abs().pow(1 / 3)

            lrs_found = True
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
                    lrs_found = False
                    print("Nesterov did not converge: lr not updated during this step.")
                    # TODO: throw warning?

            if not lrs_found:
                lrs = self.curr_lrs

            ## Additional operations on the lrs
            r = self.cfg.mom_lrs if self.step_counter > 0 else 0
            self.curr_lrs = r * self.curr_lrs + (1 - r) * lrs
            lrs = self.curr_lrs
            if self.cfg.remove_negative:
                lrs = lrs.relu()

            ## Assign lrs
            self.logs["lrs_clipped"].append(lrs)
            self.logs["curr_lrs"].append(self.curr_lrs)
            for group, lr in zip(self.param_groups, lrs, strict=False):
                group["lr"] = group["damping"] * lr.item()

        # Store logs of lrs
        self.logs["lrs"].append(
            torch.tensor(
                [group["lr"] for group in self.param_groups], device=self.device, dtype=self.dtype
            )
        )

        # Function that performs an update if necessary
        def make_step(direction):
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    for p in group["params"]:
                        p.add_(direction[i], alpha=-group["lr"])
                        i += 1

        # Perform update if self.compute_avg_Hg says so
        if update_instr.do_update:
            make_step(direction)


def create_infinite_data_loader(data_loader):
    # XXX: if the batch_size does not divide the total number of samples in
    #      the data_loader, then this may fail (possibly batches of irregular sizes)
    def f():
        for dl in itertools.repeat(data_loader):
            yield from dl

    return f

