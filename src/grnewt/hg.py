import time
import torch
from .util import ParamStructure


def compute_Hg(param_struct, full_loss, x, y, direction, *,
        noregul = False, diagonal = False, semiH = False):
    # Define useful variables
    device = param_struct.device
    dtype = param_struct.dtype
    nb_groups = param_struct.nb_groups

    # Compute gradient
    loss = full_loss(x, y)

    g_tup = param_struct.dercon(loss, direction, 0, None, detach = False)
    g = g_tup.detach()

    # Compute Hessian
    H = torch.zeros(nb_groups, nb_groups, device = device, dtype = dtype)
    order3 = torch.zeros(nb_groups, device = device, dtype = dtype)
    for i, g_i in enumerate(g_tup):
        if diagonal:
            H_i = param_struct.dercon(g_i, direction, i, i + 1, detach = False)
            H[i,i] = H_i.item()
        else:
            H_i = param_struct.dercon(g_i, direction, i, nb_groups, detach = False)

            H[i,i:] = H_i.detach()
            if not semiH:
                H[i:,i] = H_i.detach()

        # Computation of order3 (only the diagonal of the order-3 reduced derivative)
        # 2nd-order diff: differentiate g[i] w.r.t. tup_params[i]
        if noregul:
            continue

        #deriv_i = param_struct.dercon(g_i, direction, i, i + 1, detach = False)

        # 3rd-order diff
        deriv_i = param_struct.dercon(H_i[0], direction, i, i + 1, detach = True)

        # Store the result
        order3[i] = deriv_i.item()

        del H_i
        del deriv_i

    return H, g, order3

def compute_Hg_fullbatch(param_struct, full_loss, data_loader, dataset_size, direction, *,
        loader_pre_hook, noregul = False, diagonal = False):
    # Define useful variables
    device = param_struct.device
    dtype = param_struct.dtype
    nb_groups = param_struct.nb_groups

    # Compute H, g, order3
    H = torch.zeros(nb_groups, nb_groups, device = device, dtype = dtype)
    g = torch.zeros(nb_groups, device = device, dtype = dtype)
    order3 = torch.zeros(nb_groups, device = device, dtype = dtype)

    for x, y in data_loader:
        # Load samples
        x, y = loader_pre_hook(x, y)

        loss_x = lambda x_, y_: full_loss(x_, y_) * x.size(0) / dataset_size
        H_, g_, order3_ = compute_Hg(param_struct, loss_x, x, y, direction,
                noregul = noregul, diagonal = diagonal, semiH = True)

        H += H_
        g += g_
        order3 += order3_

    # H was triangular -> symmetrize it
    H = H + H.t()
    H.diagonal().mul_(.5)

    return H, g, order3

def compute_Hg_vhp(param_struct, full_loss, x, y, direction, *,
        noregul = False, diagonal = False, semiH = False):
    # Define useful variables
    device = param_struct.device
    dtype = param_struct.dtype
    nb_groups = param_struct.nb_groups

    # Compute gradient
    loss = full_loss(x, y)

    g_tup = param_struct.dercon(loss, direction, 0, None, detach = False)
    g = g_tup.detach()

    # Compute Hessian
    H = torch.zeros(nb_groups, nb_groups, device = device, dtype = dtype)
    order3 = torch.zeros(nb_groups, device = device, dtype = dtype)
    for i, g_i in enumerate(g_tup):
        if diagonal:
            H_i = param_struct.dercon(g_i, direction, i, i + 1, detach = False)
            H[i,i] = H_i.item()
        else:
            H_i = param_struct.dercon(g_i, direction, i, nb_groups, detach = False)

            H[i,i:] = H_i.detach()
            if not semiH:
                H[i:,i] = H_i.detach()

        # Computation of order3 (only the diagonal of the order-3 reduced derivative)
        # 2nd-order diff: differentiate g[i] w.r.t. tup_params[i]
        if noregul:
            continue

        #deriv_i = param_struct.dercon(g_i, direction, i, i + 1, detach = False)

        # 3rd-order diff
        deriv_i = param_struct.dercon(H_i[0], direction, i, i + 1, detach = True)

        # Store the result
        order3[i] = deriv_i.item()

        del H_i
        del deriv_i

    return H, g, order3
