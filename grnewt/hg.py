import torch

def compute_Hg(tup_params, full_loss, x, y_target, direction, *,
        param_groups = None, group_sizes = None, group_indices = None, noregul = False):
    # Define useful variables
    device = tup_params[0].device
    dtype = tup_params[0].dtype
    nb_groups = len(group_sizes)

    # Compute gradient
    loss = full_loss(x, y_target)
    grad = torch.autograd.grad(loss, tup_params, create_graph = True)

    # Reduce gradient
    g_intermediate = [(g1 * g2).sum() for g1, g2 in zip(grad, direction)]
    g_tup = tuple(sum(g_intermediate[i1:i2]) for i1, i2 in zip(group_indices[:-1], group_indices[1:]))
    g = torch.stack(g_tup).detach()

    # Compute Hessian
    H = torch.zeros(nb_groups, nb_groups, device = device, dtype = dtype)
    order3 = torch.zeros(nb_groups, device = device, dtype = dtype)
    for i, g_i in enumerate(g_tup):
        tup_params_i = [p for group in param_groups[i:] for p in group['params']]
        H_i = torch.autograd.grad(g_i, tup_params_i, retain_graph = True)

        i_direction = group_indices[i]
        H_i = torch.tensor([(g1 * g2).sum() for g1, g2 in zip(H_i, direction[i_direction:])],
                device = device, dtype = dtype)   # reduce the result to get elems of H
        H_i_split = H_i.split(group_sizes[i:])
        H_i = torch.tensor([h.sum() for h in H_i_split], device = device, dtype = dtype)

        H[i,i:] = H_i
        H[i:,i] = H_i

        # Computation of order3 (only the diagonal of the order-3 reduced derivative)
        # 2nd-order diff: differentiate g[i] w.r.t. tup_params[i]
        if noregul:
            continue
        deriv_i = torch.autograd.grad(g_i, param_groups[i]['params'], create_graph = True)
        deriv_i = sum((g1 * g2).sum() for g1, g2 in zip(deriv_i, direction[group_indices[i]:group_indices[i+1]]))   # reduce

        # 3rd-order diff
        deriv_i = torch.autograd.grad(deriv_i, param_groups[i]['params'], retain_graph = True)
        deriv_i = sum((g1 * g2).sum() for g1, g2 in zip(deriv_i, direction[group_indices[i]:group_indices[i+1]]))   # reduce

        # Add to the final result
        order3[i] += deriv_i
        del deriv_i

    return H, g, order3

