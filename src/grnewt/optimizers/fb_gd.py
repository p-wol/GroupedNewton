import torch
from torch import Tensor
from torch.optim import Optimizer


class FBGDUpdate:
    def __init__(
        self,
        model,
        loss_fn,
        train_loader,
        *,
        loader_pre_hook,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.loader_pre_hook = loader_pre_hook

        self.param_groups = [{"params": list(model.parameters())}]

    def compute_step(self):
        # Compute the dataset size
        if getattr(self.train_loader, "drop_last", False):
            raise ValueError(
                "FBGDUpdate needs every sample exactly once; train_loader has "
                "drop_last=True."
            )
        train_size = len(self.train_loader.dataset)

        # Compute full-batch gradient
        self.model.zero_grad()
        for x, y in self.train_loader:
            x, y = self.loader_pre_hook(x, y)
            curr_loss = self.loss_fn(self.model(x), y) * x.size(0) / train_size
            curr_loss.backward()

        grad = tuple(
            p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()
            )
        model.zero_grad(set_to_none=True)

        return grad
