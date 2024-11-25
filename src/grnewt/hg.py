import torch
from .util import ParamGroups


def compute_Hg(param_groups, full_loss, x, y_target, direction, *,
        noregul = False, diagonal = False, semiH = False):
    # Define useful variables
    device = param_groups.device
    dtype = param_groups.dtype
    nb_groups = param_groups.nb_groups

    # Compute gradient
    loss = full_loss(x, y_target)

    g_tup = param_groups.dercon(loss, direction, 0, None, detach = False)
    g = g_tup.detach()

    # Compute Hessian
    H = torch.zeros(nb_groups, nb_groups, device = device, dtype = dtype)
    order3 = torch.zeros(nb_groups, device = device, dtype = dtype)
    for i, g_i in enumerate(g_tup):
        if diagonal:
            H_i = param_groups.dercon(g_i, direction, i, i + 1, detach = True)
            H[i,i] = H_i.item()
        else:
            H_i = param_groups.dercon(g_i, direction, i, nb_groups, detach = True)
            H[i,i:] = H_i
            if not semiH:
                H[i:,i] = H_i

        # Computation of order3 (only the diagonal of the order-3 reduced derivative)
        # 2nd-order diff: differentiate g[i] w.r.t. tup_params[i]
        if noregul:
            continue

        deriv_i = param_groups.dercon(g_i, direction, i, i + 1, detach = False)

        # 3rd-order diff
        if not deriv_i.requires_grad:
            deriv_i.zero_()
        else:
            deriv_i = param_groups.dercon(deriv_i, direction, i, i + 1, detach = True)

        # Store the result
        order3[i] = deriv_i.item()
        del deriv_i

    return H, g, order3

def compute_Hg_fullbatch(param_groups, full_loss, data_loader, dataset_size, direction, *,
        autoencoder = False, noregul = False, diagonal = False):
    # Define useful variables
    device = param_groups.device
    dtype = param_groups.dtype
    nb_groups = param_groups.nb_groups

    # Compute H, g, order3
    H = torch.zeros(nb_groups, nb_groups, device = device, dtype = dtype)
    g = torch.zeros(nb_groups, device = device, dtype = dtype)
    order3 = torch.zeros(nb_groups, device = device, dtype = dtype)

    for x, y_target in data_loader:
        # Load samples
        x = x.to(device = device, dtype = dtype)
        if autoencoder:
            y_target = x
        else:
            y_target = y_target.to(device = device)

        loss_x = lambda x_, y_: full_loss(x_, y_) * x.size(0) / dataset_size
        H_, g_, order3_ = compute_Hg(param_groups, loss_x, x, y_target, direction,
                noregul = noregul, diagonal = diagonal, semiH = True)

        H += H_
        g += g_
        order3 += order3_

    # H was triangular -> symmetrize it
    H = H + H.t()
    H.diagonal().mul_(.5)

    return H, g, order3
