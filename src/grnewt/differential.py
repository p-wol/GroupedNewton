import copy
from itertools import combinations_with_replacement
import torch
from .util import ParamStructure

def diff_n(param_struct, order, full_loss, x, y, direction):
    # Define useful variables
    device = param_struct.device
    dtype = param_struct.dtype
    nb_groups = param_struct.nb_groups

    # Initialize tensors
    lst_results = [None] * (order + 1)

    # Compute gradient
    deriv = full_loss(x, y)
    lst_results[0] = {tuple(): deriv.detach()}

    deriv = {tuple(): param_struct.dercon(deriv, direction, 0, None, detach = False)}
    lst_results[1] = {k: v.detach() for k, v in deriv.items()}

    for d in range(2, order + 1):
        new_deriv = {}
        set_idx = [tuple(sorted(idx)) for idx in combinations_with_replacement(range(nb_groups), d - 1)]
        for idx in set_idx:
            init, last = idx[:-1], idx[-1]
            imax = last if len(init) == 0 else last - init[-1]
            new_deriv[idx] = param_struct.dercon(deriv[init][imax], direction, last, None, detach = False)
        lst_results[d] = {k: v.detach() for k, v in new_deriv.items()}
        deriv = new_deriv

    return lst_results

def diff_n_fullbatch(param_struct, order, full_loss, data_loader, dataset_size, direction,
        loader_pre_hook = lambda *args: args):
    # Define useful variables
    device = param_struct.device
    dtype = param_struct.dtype
    nb_groups = param_struct.nb_groups

    # Initialize tensors
    lst_results = None

    for x, y in data_loader:
        # Load samples
        x, y = loader_pre_hook(x, y)

        loss_x = lambda x_, y_: full_loss(x_, y_) * x.size(0) / dataset_size
        lst_results_ = diff_n(param_struct, order, loss_x, x, y, direction)

        if lst_results is None:
            lst_results = copy.deepcopy(lst_results_)
        else:
            for d in range(order + 1):
                for k, v in lst_results_[d].items():
                    lst_results[d][k].add_(lst_results_[d][k])

    return lst_results
