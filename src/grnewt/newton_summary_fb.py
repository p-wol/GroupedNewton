from torch.utils.data import DataLoader

from .config import HgCfg
from .hg import compute_Hg, compute_Hg_batched
from .ns_base import NSBase, UpdateInstructions
from .optimizers import FBGDUpdate


class NewtonSummaryFB(NSBase):
    def __init__(
        self,
        param_groups,
        full_loss,
        model,
        loss_fn,
        train_loader: DataLoader,
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
        # Create updater
        ## Create a new DataLoader to avoid problems related to double-use 
        ## of a data loader with persistent_workers=True
        self._own_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            num_workers=getattr(train_loader, "num_workers", 0),
            persistent_workers=getattr(train_loader, "num_workers", 0) > 0,
            pin_memory=getattr(train_loader, "pin_memory", False),
        )

        updater = FBGDUpdate(model, loss_fn, self._own_loader, loader_pre_hook=loader_pre_hook)

        super().__init__(
            param_groups, full_loss, train_loader, updater, loader_pre_hook=loader_pre_hook, cfg=cfg
        )

        if getattr(train_loader, "drop_last", False):
            raise ValueError(
                "NewtonSummaryFB needs every sample exactly once; train_loader has "
                "drop_last=True, which biases Hbar, gbar and order3 by three different "
                "factors."
            )
        train_size = len(train_loader.dataset)

        self.train_loader = train_loader
        self.train_size = train_size
        self.loader_pre_hook = loader_pre_hook

    def compute_avg_Hg(self, direction):
        avg_H = None
        avg_g = None
        avg_order3 = None
        for x, y in self.train_loader:
            # Compute H, g
            ## Prepare data
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
