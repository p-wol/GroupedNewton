from torch.utils.data import DataLoader

from .config import HgCfg
from .hg import compute_Hg, compute_Hg_batched
from .ns_base import NSBase, UpdateInstructions
from .optimizers import FBGDUpdate


class NewtonSummaryFB(NSBase):
    """Newton-summary optimizer whose summaries are EXACT rather than estimated.

    One `step()` is one full-batch update: it sweeps the training set once for the
    gradient and once for (Hbar, gbar, order3). `Trainer.step_train_fb` calls it once
    per epoch, outside any iteration of the training loader -- unlike every other
    optimizer, which is stepped from inside `Trainer.step_train`'s minibatch loop.
    """

    def __init__(
        self,
        param_groups,
        full_loss,
        model,
        loss_fn,
        fb_loader: DataLoader,
        *,
        loader_pre_hook,
        cfg: HgCfg,
    ):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y) = l(m(x), y)
        fb_loader: a loader over the training set that THIS OPTIMIZER OWNS, used for BOTH
            sweeps. Reasons it is a dedicated loader rather than `train_loader`:

            * Both sweeps must see the SAME batch partition. Under BatchNorm in train()
              mode the per-batch statistics make `sum_b (n_b/N) grad L_b` depend on the
              partition -- measured 1.9e-2 relative difference between batch 20 and
              batch 40, and 5.4e-2 between two shufflings of batch 20, against 2.3e-16
              in eval() mode. Computing u on one loader and Hbar on another (one of them
              shuffled) means the direction and the summaries describe different
              objectives.
            * `shuffle` is meaningless here.
            * The batch size is only a memory/throughput knob.
            * `drop_last=True` is refused below.
            * Defensively: `Trainer.step_train_fb` calls `step()` outside any iteration
              of a loader, so re-entrancy is not currently reachable. But `step()` is a
              generic entry point and every other optimizer is stepped from inside
              `Trainer.step_train`'s minibatch loop; re-entering a DataLoader with an
              active iterator and persistent_workers=True silently truncates the
              caller's epoch to a single minibatch (measured 1 batch instead of 4, no
              exception, no warning). A private loader makes that unreachable.

            `train_size` is `len(fb_loader.dataset)`; nothing else would be consistent.
        cfg: validated `optimizer.hg` node; see grnewt/config.py for every field,
             its default, and which optimizers read it. Nothing else is read from
             the config, and every field this optimizer ignores is rejected at
             composition time by grnewt.config.check_consumed.
        """
        if getattr(fb_loader, "drop_last", False):
            raise ValueError(
                "NewtonSummaryFB needs every sample exactly once; fb_loader has "
                "drop_last=True, which biases Hbar, gbar and order3 by three different "
                "factors."
            )

        updater = FBGDUpdate(model, loss_fn, fb_loader, loader_pre_hook=loader_pre_hook)

        # data_loader=None: this optimizer draws no minibatch of its own.
        super().__init__(
            param_groups, full_loss, None, updater, loader_pre_hook=loader_pre_hook, cfg=cfg
        )

        self.fb_loader = fb_loader
        self.train_size = len(fb_loader.dataset)

    def compute_avg_Hg(self, direction):
        avg_H = None
        avg_g = None
        avg_order3 = None
        for x, y in self.fb_loader:
            # Compute H, g
            ## Prepare data
            x, y = self.loader_pre_hook(x, y)
            loss = self.full_loss(x, y)

            ## Compute H, g, order3
            cp_kwargs = {"noregul": self.cfg.noregul, "diagonal": self.cfg.diagonal}
            if self.cfg.hg_batched:
                cp_Hg = compute_Hg_batched
                cp_kwargs["chunk_size"] = self.cfg.hg_batched_chunk
            else:
                cp_Hg = compute_Hg

            H, g, order3 = cp_Hg(
                self.param_struct,
                loss,
                direction,
                **cp_kwargs,
            )

            H.mul_(x.size(0) / self.train_size)
            g.mul_(x.size(0) / self.train_size)
            order3.mul_(x.size(0) / self.train_size)

            if avg_H is None:
                avg_H = H.detach()
                avg_g = g.detach()
                avg_order3 = order3.detach()
            else:
                avg_H.add_(H.detach())
                avg_g.add_(g.detach())
                avg_order3.add_(order3.detach())

        return avg_H, avg_g, avg_order3, UpdateInstructions(recompute_lrs=True, do_update=True)
