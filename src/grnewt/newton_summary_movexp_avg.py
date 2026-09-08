import torch
from torch.utils.data import DataLoader

from .config import HgCfg
from .hg import compute_Hg, compute_Hg_batched
from .ns_base import NSBase, UpdateInstructions


class NewtonSummaryMovexpAvg(NSBase):
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
        super().__init__(
            param_groups, full_loss, data_loader, updater, loader_pre_hook=loader_pre_hook, cfg=cfg
        )

        self.movavg = cfg.movexp_avg.movavg
        self.dct_HgD_avgs = {k: None for k in ["H", "g", "D"]}

    def update_movexp_avg(self, H, g, order3):
        """
        Updates the attribute dct_HgD_avgs.
        """
        dct_HgD = {"H": H, "g": g, "D": order3}

        # Init the moving averages
        if self.step_counter == 0:
            for key, curr in dct_HgD.items():
                self.dct_HgD_avgs[f"{key}"] = torch.zeros_like(curr)

        # Update the moving averages
        r = self.movavg
        for key, _curr in dct_HgD.items():
            self.dct_HgD_avgs[f"{key}"] = (1 - r) * self.dct_HgD_avgs[f"{key}"] + r * dct_HgD[
                f"{key}"
            ]

        # Remove the bias (Adam-like)
        t = self.step_counter // self.cfg.period_hg
        for key, _curr in dct_HgD.items():
            dct_HgD[f"{key}"] = self.dct_HgD_avgs[f"{key}"] / (1 - (1 - r) ** (t + 1))

        # Return the H, g, order3 to use
        H = dct_HgD["H"]
        g = dct_HgD["g"]
        order3 = dct_HgD["D"]

        return H, g, order3

    def compute_avg_Hg(self, direction):
        # If we do not update H, g, order3 and lrs: just move forward
        if self.step_counter % self.cfg.period_hg != 0:
            return None, None, None, UpdateInstructions(recompute_lrs=False, do_update=True)

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

        avg_H, avg_g, avg_order3 = self.update_movexp_avg(H, g, order3)

        return avg_H, avg_g, avg_order3, UpdateInstructions(recompute_lrs=True, do_update=True)
