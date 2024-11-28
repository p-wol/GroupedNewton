import torch

def fullbatch_gradient(param_groups, loss_fn, model, train_loader, train_size, *,
        loader_pre_hook):
    # Compute full-batch gradient
    model.zero_grad()
    for x, y in train_loader:
        x, y = loader_pre_hook(x, y)

        y_hat = model(x)
        curr_loss = loss_fn(y_hat, y) * x.size(0) / train_size
        curr_loss.backward()
    
    grad = tuple(p.grad.clone() for p in param_groups.tup_params)
    model.zero_grad()

    return grad
