import torch

def compute_Hg(tup_params, full_loss, x, y_target, direction, *,
        param_groups = None, group_sizes = None, group_indices = None, noregul = False,
        diagonal = False):
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
        if not diagonal:
            tup_params_i = [p for group in param_groups[i:] for p in group['params']]
            H_i = torch.autograd.grad(g_i, tup_params_i, retain_graph = True)

            i_direction = group_indices[i]
            H_i = torch.tensor([(g1 * g2).sum() for g1, g2 in zip(H_i, direction[i_direction:])],
                    device = device, dtype = dtype)   # reduce the result to get elems of H
            H_i_split = H_i.split(group_sizes[i:])
            H_i = torch.tensor([h.sum() for h in H_i_split], device = device, dtype = dtype)

            H[i,i:] = H_i
            H[i:,i] = H_i
        else:
            H_i = torch.autograd.grad(g_i, param_groups[i]['params'], create_graph = True, materialize_grads = True)
            H_i = sum((g1 * g2).sum() for g1, g2 in zip(H_i, direction[group_indices[i]:group_indices[i+1]]))   # reduce
            H[i,i] = H_i.item()

        # Computation of order3 (only the diagonal of the order-3 reduced derivative)
        # 2nd-order diff: differentiate g[i] w.r.t. tup_params[i]
        if noregul:
            continue

        if not diagonal:
            deriv_i = torch.autograd.grad(g_i, param_groups[i]['params'], create_graph = True, materialize_grads = True)
            deriv_i = sum((g1 * g2).sum() for g1, g2 in zip(deriv_i, direction[group_indices[i]:group_indices[i+1]]))   # reduce
        else:
            deriv_i = H_i

        # 3rd-order diff
        if not deriv_i.requires_grad:
            deriv_i.zero_()
        else:
            deriv_i = torch.autograd.grad(deriv_i, param_groups[i]['params'], retain_graph = True, materialize_grads = True)
            #deriv_i = tuple(p if p is not None else torch.tensor(0, dtype = dtype, device = device) for p in deriv_i)

            deriv_i = sum((g1 * g2).sum() for g1, g2 in zip(deriv_i, direction[group_indices[i]:group_indices[i+1]]))   # reduce

        # Add to the final result
        order3[i] += deriv_i
        del deriv_i

    return H, g, order3

def compute_Hg_fullbatch(tup_params, full_loss, data_loader, dataset_size, direction, *,
        param_groups = None, group_sizes = None, group_indices = None, noregul = False, 
        autoencoder = False):
    # Define useful variables
    device = tup_params[0].device
    dtype = tup_params[0].dtype
    nb_groups = len(group_sizes)

    # Compute H, g, order3
    H = torch.zeros(nb_groups, nb_groups, device = device, dtype = dtype)
    g = torch.zeros(nb_groups, device = device, dtype = dtype)
    order3 = torch.zeros(nb_groups, device = device, dtype = dtype)

    for x, y in data_loader:
        # Load samples
        x = x.to(device = device, dtype = dtype)
        if autoencoder:
            y = x
        else:
            y = y.to(device = device)

        # Compute gradient
        curr_loss = full_loss(x, y) * x.size(0) / dataset_size
        gx = torch.autograd.grad(curr_loss, tup_params, create_graph = True)

        # Reduce gradient
        gx_intermediate = [(g1 * g2).sum() for g1, g2 in zip(gx, direction)]
        gx_tup = tuple(sum(gx_intermediate[i1:i2]) for i1, i2 in zip(group_indices[:-1], group_indices[1:]))
        g += torch.stack(gx_tup).detach()

        # Compute H and order3 from g
        for i, gx_i in enumerate(gx_tup):
            tup_params_i = [p for group in param_groups[i:] for p in group['params']]
            Hx_i = torch.autograd.grad(gx_i, tup_params_i, retain_graph = True)

            i0 = group_indices[i]
            Hx_i = torch.tensor([(g1 * g2).sum() for g1, g2 in zip(Hx_i, direction[i0:])],
                    device = device, dtype = dtype)   # reduce the result to get elems of H
            Hx_i = tuple(sum(Hx_i[i1-i0:i2-i0]) for i1, i2 in zip(group_indices[i:-1], group_indices[i+1:]))

            H[i,i:] += torch.stack(Hx_i).detach()

            if not noregul:
                # Computation of order3 (only the diagonal of the order-3 reduced derivative)
                # 2nd-order diff: differentiate g[i] w.r.t. tup_params[i]
                deriv_i = torch.autograd.grad(gx_i, param_groups[i]['params'], create_graph = True)
                deriv_i = sum((g1 * g2).sum() for g1, g2 in zip(deriv_i, direction[group_indices[i]:group_indices[i+1]]))   # reduce

                # 3rd-order diff
                deriv_i = torch.autograd.grad(deriv_i, param_groups[i]['params'], retain_graph = True)
                deriv_i = sum((g1 * g2).sum() for g1, g2 in zip(deriv_i, direction[group_indices[i]:group_indices[i+1]]))   # reduce

                # Add to the final result
                order3[i] += deriv_i
                del deriv_i

    # H was triangular -> symmetrize it
    H = H + H.t()
    H.diagonal().mul_(.5)

    return H, g, order3
