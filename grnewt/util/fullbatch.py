import torch

def fullbatch_gradient(model, final_loss, tup_params, train_loader, train_size, *,
        autoencoder = False):
    # Define useful variables
    p = next(iter(model))
    device = p.device
    dtype = p.dtype

    # Compute full-batch_gradient
    model.zero_grad()
    for x, y in train_loader:
        x = x.to(device = device, dtype = dtype)
        if autoencoder:
            y = x
        else:
            y = y.to(device = device)

        y_hat = model(x)
        curr_loss = final_loss(y_hat, y) * x.size(0) / train_size
        curr_loss.backward()
    
    grad = tuple(p.grad.clone() for p in tup_params)
    model.zero_grad()

    return grad
