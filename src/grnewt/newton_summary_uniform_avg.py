import itertools

import torch
from torch.utils.data import DataLoader

from .config import HgCfg
from .hg import compute_Hg, compute_Hg_batched
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure


def increment_step(func):
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.step_counter += 1
        return ret

    return wrapper


class NewtonSummaryUniformAvg(torch.optim.Optimizer):
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

        self.dct_HgD_avgs = {k: None for k in ["H_use", "H_up", "g_use", "g_up", "D_use", "D_up"]}

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

    def update_uniform_avg(self, H, g, order3):
        """
        Updates the attribute dct_HgD_avgs.
        """
        dct_HgD = {"H": H, "g": g, "D": order3}

        # Compute the time step of the method "uniform average"
        t = (self.step_counter // self.cfg.period_hg) % self.cfg.uniform_avg.period

        # Time to replace the current moving average
        if t == 0:
            # At init, set the moving averages to zero.
            #
            # FIX (2026-08-21): both assignments below used to store `curr`
            # ITSELF.  `_use` and `_up` then aliased each other AND the incoming
            # sample, so the update loop below applied `mul_((n-1)/n)` twice to
            # one tensor -- and with n = 1 the first `mul_(0)` zeroed `curr`
            # before `add_((1/n) * curr)` could read it.  Measured effect: the
            # returned average was exactly 0 at step 0, the two accumulators
            # stayed aliased for the whole first period, and every later period
            # silently dropped its first sample while `_up` was biased low by
            # (P-1)/P.  Since H, g and D all carry the SAME wrong weight W, the
            # net effect on the step is lambda_eff = W^2 * lambda_int (verified);
            # in the reference LeNet/CIFAR configuration W cycled through
            # {3/4, 4/5, 5/6}, i.e. lambda_eff/lambda_int in {0.56, 0.64, 0.69}.
            if self.step_counter == 0:
                for key, curr in dct_HgD.items():
                    self.dct_HgD_avgs[f"{key}_up"] = torch.zeros_like(curr)

            # Replace the current moving average by the next one and set the other to zero
            for key, curr in dct_HgD.items():
                self.dct_HgD_avgs[f"{key}_use"] = self.dct_HgD_avgs[f"{key}_up"]
                self.dct_HgD_avgs[f"{key}_up"] = torch.zeros_like(curr)

        # During the first period, both averages "use" and "up" are updated with the same coefficient
        offset_use = (
            0
            if self.step_counter // self.cfg.period_hg < self.cfg.uniform_avg.period
            else self.cfg.uniform_avg.period
        )

        # Set up the time coefficients
        tt_use = offset_use + t + 1
        tt_up = t + 1

        # Update the moving averages.
        # `curr / n` is materialized BEFORE the in-place `mul_`, so the two
        # accumulators can never read a tensor that the other has just mutated
        # even if a future refactor reintroduces sharing.
        for key1, curr in dct_HgD.items():
            for key2, n in zip(["use", "up"], [tt_use, tt_up], strict=False):
                key = f"{key1}_{key2}"
                contrib = curr / n
                self.dct_HgD_avgs[key].mul_((n - 1) / n).add_(contrib)

        # Return the H, g, order3 to use
        H = self.dct_HgD_avgs["H_use"]
        g = self.dct_HgD_avgs["g_use"]
        order3 = self.dct_HgD_avgs["D_use"]

        return H, g, order3

    @increment_step
    def step(self):
        # Perform update only if warm-up phase has ended
        warmup_ended = self.step_counter // self.cfg.period_hg >= self.cfg.uniform_avg.warmup

        # Function that performs an update if necessary
        def make_step(direction):
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    for p in group["params"]:
                        p.add_(direction[i], alpha=-group["lr"])
                        i += 1

        # If we do not update H, g, order3 and lrs: just move forward
        if self.step_counter % self.cfg.period_hg != 0:
            # If warm-up phase has ended, perform update (else, do nothing)
            if warmup_ended:
                direction = self.param_struct.reindex(
                    self.updater.compute_step(), self._dir_perm)
                make_step(direction)
            return

        # If we update H, g, order3 and lrs: first compute the direction
        direction = self.param_struct.reindex(
            self.updater.compute_step(), self._dir_perm)

        # Compute H, g
        ## Prepare data
        x, y = next(self.dl_iter)
        x, y = self.loader_pre_hook(x, y)

        ## Compute H, g, order3
        cp_kwargs = {"noregul": self.cfg.noregul,
                     "diagonal": self.cfg.diagonal}
        if self.cfg.hg_batched:
            cp_Hg = compute_Hg_batched
            cp_kwargs["chunk_size"] = self.cfg.hg_batched_chunk
        else:
            cp_Hg = compute_Hg

        H, g, order3 = cp_Hg(
            self.param_struct,
            self.full_loss,
            x,
            y,
            direction,
            **cp_kwargs,
        )

        # Update the averages of H, g, order3
        H, g, order3 = self.update_uniform_avg(H, g, order3)

        order3_ = order3.abs().pow(1 / 3)

        ## Store logs of H, g, order3
        self.logs["H"].append(H)
        self.logs["g"].append(g)
        self.logs["order3"].append(order3)

        # If we are still in the warm-up phase, do not compute the lrs and do not update
        if not warmup_ended:
            return

        # Compute lrs
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

        # Perform update
        make_step(direction)


def create_infinite_data_loader(data_loader):
    # XXX: if the batch_size does not divide the total number of samples in
    #      the data_loader, then this may fail (possibly batches of irregular sizes)
    def f():
        for dl in itertools.repeat(data_loader):
            yield from dl

    return f
