from itertools import product, repeat, permutations, combinations, combinations_with_replacement
import torch
from .util import ParamGroups

def diff_n(param_groups, order, full_loss, x, y_target, direction):
    # Define useful variables
    device = param_groups.device
    dtype = param_groups.dtype
    nb_groups = param_groups.nb_groups

    # Initialize tensors
    lst_results = [None] * (order+1)
    for d in range(1, order+1):
        lst_results[d] = torch.zeros(*([nb_groups] * d), device = device, dtype = dtype)

    # Compute gradient
    loss = full_loss(x, y_target)
    lst_results[0] = loss.detach()

    deriv = param_groups.dercon(loss, direction, 0, None, detach = False)
    lst_results[1] = deriv.detach()

    for d in range(2, order + 1):
        new_result = torch.zeros(*([nb_groups] * d), device = device, dtype = dtype)
        set_idx = [sorted(idx) for idx in combinations_with_replacement(range(nb_groups), d-1)]
        for idx in set_idx:
            new_der = param_groups.dercon(deriv[*idx], direction, idx[-1], None, detach = False)
            new_result[*idx, idx[-1]:] = new_der
        lst_results[d] = new_result.detach()
        deriv = new_result

    return lst_results

def diff_n_fullbatch(param_groups, order, full_loss, data_loader, dataset_size, direction,
        autoencoder = False):
    # Define useful variables
    device = param_groups.device
    dtype = param_groups.dtype
    nb_groups = param_groups.nb_groups

    # Initialize tensors
    lst_results = [None] * (order + 1)
    for d in range(1, order + 1):
        lst_results[d] = torch.zeros(*([nb_groups] * d), device = device, dtype = dtype)
    lst_results[0] = torch.tensor(0., device = device, dtype = dtype)

    for x, y_target in data_loader:
        # Load samples
        x = x.to(device = device, dtype = dtype)
        if autoencoder:
            y_target = x
        else:
            y_target = y_target.to(device = device)

        loss_x = lambda x_, y_: full_loss(x_, y_) * x.size(0) / dataset_size
        lst_results_ = diff_n(param_groups, order, loss_x, x, y_target, direction)

        for d in range(order + 1):
            lst_results[d] += lst_results_[d]

    return lst_results
