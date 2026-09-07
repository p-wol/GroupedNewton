
import torch
from torch.utils.data import DataLoader

from .config import HgCfg
from .hg import compute_Hg, compute_Hg_batched
from .ns_base import NSBase, UpdateInstructions


class NewtonSummaryUniformAvg(NSBase):
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
        super().__init__(param_groups, full_loss, data_loader, updater,
                loader_pre_hook=loader_pre_hook, cfg=cfg)

        self.dct_HgD_avgs = {k: None for k in ["H_use", "H_up", "g_use", "g_up", "D_use", "D_up"]}

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

    def compute_avg_Hg(self, direction):
        # Perform update only if warm-up phase has ended
        warmup_ended = self.step_counter // self.cfg.period_hg >= self.cfg.uniform_avg.warmup

        # If we do not update H, g, order3 and lrs: just move forward
        if self.step_counter % self.cfg.period_hg != 0:
            # If warm-up phase has ended, perform update (else, do nothing)
            return None, None, None, UpdateInstructions(recompute_lrs=False, do_update=warmup_ended)

        # Compute H, g
        ## Prepare data
        x, y = next(self.dl_iter)
        x, y = self.loader_pre_hook(x, y)

        ## Compute H, g, order3
        cp_kwargs = {"noregul": self.cfg.noregul, "diagonal": self.cfg.diagonal}
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

        return H, g, order3, UpdateInstructions(recompute_lrs=warmup_ended, do_update=warmup_ended)

