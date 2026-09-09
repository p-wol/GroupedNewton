import torch
from torch.utils.data import DataLoader

from .config import HgCfg
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure


def increment_step(func):
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.step_counter += 1
        return ret

    return wrapper


class NewtonSummary(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        loss,
        updater,
        *,
        cfg: HgCfg,
    ):
        """
        param_groups: param_groups of the model
        loss: current loss (result of a PyTorch computation involving the params)
        cfg: validated `optimizer.hg` node; see grnewt/config.py for every field,
             its default, and which optimizers read it. Nothing else is read from
             the config, and every field this optimizer ignores is rejected at
             composition time by grnewt.config.check_consumed.
        """
        self.loss = loss
        self.updater = updater
        self.cfg = cfg
        self.curr_lrs = 0
        self.dir_norm = None

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

    def normalize_dirs_(self, direction):
        norm = self.param_struct.squared_norm(direction).sqrt()
        norm = self.param_struct.expand_src_as_params(norm)

        for d, n in zip(direction, norm, strict=True):
            if n > 0:
                d.div_(n)

        return norm

    def compute_avg_Hg(self, direction):
        # Compute H, g
        ## Compute H, g, order3
        cp_kwargs = {"noregul": self.cfg.noregul, "diagonal": self.cfg.diagonal}
        if self.cfg.hg_batched:
            cp_Hg = compute_Hg_batched
            cp_kwargs["chunk_size"] = self.cfg.hg_batched_chunk
        else:
            cp_Hg = compute_Hg

        H, g, order3 = cp_Hg(
            self.param_struct,
            self.loss,
            direction,
            **cp_kwargs,
        )

        return H.detach(), g.detach(), order3.detach()

    @increment_step
    def step(self):
        # Function that performs an update
        def make_step(direction):
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    for p in group["params"]:
                        p.add_(direction[i], alpha=-group["lr"])
                        i += 1

        # Compute the direction
        direction = self.param_struct.reindex(self.updater.compute_step(), self._dir_perm)

        # Normalize if required
        if self.cfg.normalize_dirs:
            self.dir_norm = self.normalize_dirs_(direction)

        # Compute the averages of H, g, order3
        # H, g, order3, update_instr = self.compute_avg_Hg(direction_normed)
        H, g, order3, update_instr = self.compute_avg_Hg(direction)

        # Store logs of H, g, order3
        self.logs["H"].append(H)
        self.logs["g"].append(g)
        self.logs["order3"].append(order3)

        # Compute order3_
        order3_ = order3.abs().pow(1 / 3)

        lrs_found = True
        if self.cfg.noregul:
            # no regularization
            lrs = torch.linalg.solve(H, g)
        elif not self.cfg.nesterov.use:
            # with regularization, but no Nesterov cubic regul
            # => Tikhonov regularization
            regul_H = self.cfg.ridge * torch.eye(H.size(0), dtype=self.dtype, device=self.device)
            lrs = torch.linalg.solve(H + regul_H, g)
        else:
            # regularization with Nesterov cubic
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

        ### To finish: perform update if necessary ###
        make_step(direction)
