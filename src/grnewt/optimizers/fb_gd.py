import torch
from torch import Tensor
from torch.optim import Optimizer


class FBGDUpdate:
    def __init__(
        self,
        model,
        loss_fn,
        train_loader,
        train_size,
        *,
        loader_pre_hook,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.train_size = train_size
        self.loader_pre_hook = loader_pre_hook

        self.param_groups = [{'params': [p for p in model.parameters()]}]

    def compute_step(self):
        return fullbatch_gradient(
                self.loss_fn, 
                self.model, 
                self.train_loader, 
                self.train_size, 
                loader_pre_hook=self.loader_pre_hook)

def fullbatch_gradient(loss_fn, model, train_loader, train_size, *, loader_pre_hook):
    # Compute full-batch gradient
    model.zero_grad()
    for x, y in train_loader:
        x, y = loader_pre_hook(x, y)

        y_hat = model(x)
        curr_loss = loss_fn(y_hat, y) * x.size(0) / train_size
        curr_loss.backward()

    grad = tuple(p.grad.clone() for p in model.parameters())
    model.zero_grad()

    return grad
