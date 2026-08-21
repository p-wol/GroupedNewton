import itertools

import torch
from torch.utils.data import DataLoader

from .config import HgCfg
from .hg import compute_Hg, compute_Hg_batched
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure
from .ns_base import NSBase, UpdateInstructions


class NewtonSummaryStaticAvg(NSBase):
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

        self.nsamples = cfg.static_avg.nsamples
        print("### WARNING ### the default config for nsamples is not overriden: to investigate")
        print(self.nsamples)

    def compute_avg_Hg(self, direction):
        # If we do not update H, g, order3 and lrs: just move forward
        if self.step_counter % self.cfg.period_hg != 0:
            return None, None, None, UpdateInstructions(recompute_lrs=False, do_update=True)

        avg_H = None
        avg_g = None
        avg_order3 = None
        for i in range(self.nsamples):
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

            if avg_H is None:
                avg_H = H.detach()
                avg_g = g.detach()
                avg_order3 = order3.detach()
            else:
                avg_H.add_(H.detach())
                avg_g.add_(g.detach())
                avg_order3.add_(order3.detach())

        avg_H.div_(self.nsamples)
        avg_g.div_(self.nsamples)
        avg_order3.div_(self.nsamples)

        return avg_H, avg_g, avg_order3, UpdateInstructions(recompute_lrs=True, do_update=True)


