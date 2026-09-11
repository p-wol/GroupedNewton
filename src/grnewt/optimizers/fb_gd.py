import torch

from .updater import Updater


class FBGDUpdate(Updater):
    """Updater whose "step" is the exact full-batch gradient.

    Interface expected by NSBase: `.param_groups` (read once, to build the permutation
    from this producer's order to ParamStructure order) and `.compute_step()` returning
    one tensor per parameter, in `.param_groups` order. Stateless: the tensors are fresh
    on every call, so NSBase may normalize them in place.

    `loader` must be a loader nobody else is iterating -- see NewtonSummaryFB.
    """

    def __init__(self, model, loss_fn, loader, *, loader_pre_hook):
        self.model = model
        self.loss_fn = loss_fn
        self.loader = loader
        self.loader_pre_hook = loader_pre_hook

        # Checked once, not on every step: `len(dataset)` is cheap but the invariant is
        # a property of the loader, so a violation should be reported at construction.
        if getattr(loader, "drop_last", False):
            raise ValueError(
                "FBGDUpdate needs every sample exactly once; loader has drop_last=True."
            )
        self.train_size = len(loader.dataset)

        self.last_loss_avg = None

        defaults = dict()
        super().__init__(model.parameters(), defaults)

    def compute_step(self):
        self.model.zero_grad(set_to_none=True)
        loss_avg = torch.zeros(
            (),
            dtype=self.param_groups[0]["params"][0].dtype,
            device=self.param_groups[0]["params"][0].device,
        )
        for x, y in self.loader:
            x, y = self.loader_pre_hook(x, y)
            loss = self.loss_fn(self.model(x), y) * x.size(0) / self.train_size
            loss.backward()
            loss_avg += loss.detach()

        grad = tuple(
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in self.model.parameters()
        )
        self.model.zero_grad(set_to_none=True)

        self.last_loss_avg = loss_avg.item()
        return grad
