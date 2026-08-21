"""
hg_batched_v2.py -- batched drop-in for grnewt.hg.compute_Hg.

Written against the actual grnewt API (ParamStructure / dercon), NOT EXECUTED
(no torch in the authoring environment). Run `test_matches_reference()` first.

Same signature and same return convention as grnewt.hg.compute_Hg:
    H[i, j]   = u_i^T H_ij u_j
    g[i]      = <u_i, grad_i>
    order3[i] = D^3 L[u_i, u_i, u_i]

What changes vs. the original:
  * the `for i in range(nb_groups)` loop is chunked and vmapped via
    is_grads_batched, which keeps the existing triangular input restriction
    (inputs = groups i..S-1) that torch.func.jvp would throw away;
  * order3 is obtained from the SAME batched second-order graph;
  * no per-iteration .item(): H and order3 are assembled on device and
    synchronised once.

Tunable: chunk_size trades memory for launch efficiency. Peak second-order
graph memory is ~ chunk_size * (activation memory). Start at 4 and raise.
"""

from typing import Optional

import torch

def compute_Hg(
    param_struct, full_loss, x, y, direction, *, noregul=False, diagonal=False, semiH=False
):
    # Define useful variables
    device = param_struct.device
    dtype = param_struct.dtype
    nb_groups = param_struct.nb_groups

    # Compute gradient
    loss = full_loss(x, y)

    g_tup = param_struct.dercon(loss, direction, 0, None, detach=False)
    g = g_tup.detach()

    # Compute the projections of the derivatives
    H_list = [None] * nb_groups
    order3_list = [None] * nb_groups
    for i, g_i in enumerate(g_tup):
        if diagonal:
            H_i = param_struct.dercon(g_i, direction, i, i + 1, detach=False)
            H_list[i] = H_i.detach()
        else:
            H_i = param_struct.dercon(g_i, direction, i, nb_groups, detach=False)
            H_list[i] = H_i.detach()

        # Computation of order3 (only the diagonal of the order-3 reduced derivative)
        # 2nd-order diff: differentiate g[i] w.r.t. tup_params[i]
        if noregul:
            continue

        # deriv_i = param_struct.dercon(g_i, direction, i, i + 1, detach = False)

        # 3rd-order diff
        deriv_i = param_struct.dercon(H_i[0], direction, i, i + 1, detach=True)

        # Store the result
        order3_list[i] = deriv_i.detach().squeeze()

    # Build H
    H = torch.zeros(nb_groups, nb_groups, device=device, dtype=dtype)
    for i, H_i in enumerate(H_list):
        if diagonal:
            H[i, i] = H_i.detach()
        else:
            H[i, i:] = H_i.detach()
            if not semiH:
                H[i:, i] = H_i.detach()

    # Build order3
    order3 = torch.stack(order3_list)

    return H, g, order3

def compute_Hg_fullbatch(
    param_struct,
    full_loss,
    data_loader,
    dataset_size,
    direction,
    *,
    loader_pre_hook,
    noregul=False,
    diagonal=False,
):
    # Define useful variables
    device = param_struct.device
    dtype = param_struct.dtype
    nb_groups = param_struct.nb_groups

    # Compute H, g, order3
    H = torch.zeros(nb_groups, nb_groups, device=device, dtype=dtype)
    g = torch.zeros(nb_groups, device=device, dtype=dtype)
    order3 = torch.zeros(nb_groups, device=device, dtype=dtype)

    for x, y in data_loader:
        # Load samples
        x, y = loader_pre_hook(x, y)

        loss_x = lambda x_, y_: full_loss(x_, y_) * x.size(0) / dataset_size
        H_, g_, order3_ = compute_Hg(
            param_struct, loss_x, x, y, direction, noregul=noregul, diagonal=diagonal, semiH=True
        )

        H += H_
        g += g_
        order3 += order3_

    # H was triangular -> symmetrize it
    H = H + H.t()
    H.diagonal().mul_(0.5)

    return H, g, order3

# ---------------------------------------------------------------------------
# optional fused contraction: out[k, s] = sum_{n in group s} R[k, n] * u[n]
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _seg_contract_kernel(
        R_ptr,
        u_ptr,
        out_ptr,
        tile_seg_ptr,
        stride_rk,
        stride_rn,
        stride_ok,
        N,
        BLOCK: tl.constexpr,
    ):
        k = tl.program_id(0)
        t = tl.program_id(1)
        offs = t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        r = tl.load(R_ptr + k * stride_rk + offs * stride_rn, mask=mask, other=0.0)
        uu = tl.load(u_ptr + offs, mask=mask, other=0.0)
        acc = tl.sum((r * uu).to(tl.float32), axis=0)
        s = tl.load(tile_seg_ptr + t)
        tl.atomic_add(out_ptr + k * stride_ok + s, acc)

except Exception:  # pragma: no cover
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# batched compute_Hg
# ---------------------------------------------------------------------------


def _contract(param_struct, batched, direction, start: int, end: int, k: int) -> torch.Tensor:
    """
    batched: tuple of tensors, batched[t] has shape (k, *shape of param start..end).
    Returns (k, end - start): per-group contraction against `direction`.
    """
    dirs = param_struct.select_params(src=direction, start=start, end=end)
    gi = param_struct.group_indices
    i0 = gi[start]

    per_tensor = [
        (b.reshape(k, -1) * d.reshape(1, -1)).sum(dim=1)  # (k,)
        for b, d in zip(batched, dirs)
    ]
    cols = [
        torch.stack(per_tensor[i1 - i0 : i2 - i0], dim=0).sum(dim=0)
        for i1, i2 in zip(gi[start:end], gi[start + 1 : end + 1])
    ]
    return torch.stack(cols, dim=1)  # (k, end-start)


def compute_Hg_batched(
    param_struct,
    full_loss,
    x,
    y,
    direction,
    *,
    noregul: bool = False,
    diagonal: bool = False,
    semiH: bool = False,
    chunk_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = param_struct.device, param_struct.dtype
    S = param_struct.nb_groups
    chunk_size = S if chunk_size == -1 else chunk_size

    loss = full_loss(x, y)
    g_tup = param_struct.dercon(loss, direction, 0, None, detach=False)  # (S,), with graph
    g = g_tup.detach()

    if not g_tup.requires_grad:
        z = torch.zeros(S, S, device=device, dtype=dtype)
        return z, g, torch.zeros(S, device=device, dtype=dtype)

    H = torch.zeros(S, S, device=device, dtype=dtype)
    order3 = torch.zeros(S, device=device, dtype=dtype)
    need_o3 = not noregul

    for lo in range(0, S, chunk_size):
        hi = min(lo + chunk_size, S)
        k = hi - lo

        end = lo + 1 if diagonal else S
        # keep the triangular restriction: differentiate only w.r.t. groups lo..end-1
        inputs = param_struct.select_params(start=lo, end=end)

        # one-hot selector over the S outputs of g_tup, batched over the chunk
        E = torch.zeros(k, S, device=device, dtype=g_tup.dtype)
        E[torch.arange(k, device=device), torch.arange(lo, hi, device=device)] = 1.0

        rows = torch.autograd.grad(
            g_tup,
            inputs,
            grad_outputs=E,
            is_grads_batched=True,
            create_graph=need_o3,
            retain_graph=True,
            materialize_grads=True,
        )

        blk = _contract(param_struct, rows, direction, lo, end, k)  # (k, end-lo)

        if diagonal:
            H[torch.arange(lo, hi), torch.arange(lo, hi)] = blk[:, 0].detach()
        else:
            # blk[j] holds H[lo+j, lo:]; only columns >= lo+j belong to the upper triangle
            for j in range(k):
                H[lo + j, lo + j :] = blk[j, j:].detach()
                if not semiH:
                    H[lo + j :, lo + j] = blk[j, j:].detach()

        if need_o3:
            # diag_chunk[j] = H[lo+j, lo+j], still attached to the graph
            diag_chunk = blk[torch.arange(k, device=device), torch.arange(k, device=device)]
            own = param_struct.select_params(start=lo, end=hi)
            I = torch.eye(k, device=device, dtype=diag_chunk.dtype)
            d3 = torch.autograd.grad(
                diag_chunk,
                own,
                grad_outputs=I,
                is_grads_batched=True,
                retain_graph=True,
                materialize_grads=True,
            )
            full = _contract(param_struct, d3, direction, lo, hi, k)  # (k, k)
            order3[lo:hi] = full[
                torch.arange(k, device=device), torch.arange(k, device=device)
            ].detach()

        del rows, blk

    return H, g, order3
