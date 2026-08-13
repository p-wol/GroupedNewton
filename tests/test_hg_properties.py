"""
Tests for compute_Hg that do NOT re-implement it with autograd.

The existing tests compare compute_Hg against another autograd computation,
so a shared misuse of autograd passes both. These compare against:
  * finite differences of the loss along the direction (the definition), and
  * a closed-form quadratic oracle,
and check algebraic invariants that hold for any correct implementation.
"""

import pytest
import torch

from conftest import PARTITION_BUILDERS, Quadratic
from grnewt import ParamStructure, compute_Hg
from grnewt import partition as build_partition

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _masked(direction, param_struct, s):
    """P_s u : the direction restricted to group s, zero elsewhere."""
    gi = param_struct.group_indices
    return tuple(
        d.clone() if gi[s] <= i < gi[s + 1] else torch.zeros_like(d)
        for i, d in enumerate(direction)
    )


def _phi(param_struct, full_loss, x, y, v, t):
    """L(theta + t v), evaluated without disturbing the parameters."""
    with torch.no_grad():
        for p, d in zip(param_struct.tup_params, v):
            p.add_(d, alpha=t)
    try:
        with torch.no_grad():
            return full_loss(x, y).item()
    finally:
        with torch.no_grad():
            for p, d in zip(param_struct.tup_params, v):
                p.add_(d, alpha=-t)


def _derivs_fd(param_struct, full_loss, x, y, v, eps=1e-3):
    """Central differences of phi(t) = L(theta + t v) at t = 0, orders 1..3."""
    def f(t):
        return _phi(param_struct, full_loss, x, y, v, t)

    f0 = f(0.0)
    fp1, fm1 = f(eps), f(-eps)
    fp2, fm2 = f(2 * eps), f(-2 * eps)
    d1 = (fp1 - fm1) / (2 * eps)
    d2 = (fp1 - 2 * f0 + fm1) / eps**2
    d3 = (fp2 - 2 * fp1 + 2 * fm1 - fm2) / (2 * eps**3)
    return d1, d2, d3


# --------------------------------------------------------------------------
# 1. the definition
# --------------------------------------------------------------------------


def test_gbar_and_diagonal_match_finite_differences(
    mlp, batch, mse, param_struct, direction
):
    """gbar[s], Hbar[s,s] and order3[s] are the 1st/2nd/3rd derivatives of
    t -> L(theta + t P_s u) at t = 0."""
    x, y = batch(mlp)
    full_loss = mse(mlp)
    H, g, order3 = compute_Hg(param_struct, full_loss, x, y, direction)

    for s in range(param_struct.nb_groups):
        v = _masked(direction, param_struct, s)
        d1, d2, d3 = _derivs_fd(param_struct, full_loss, x, y, v)
        assert g[s].item() == pytest.approx(d1, rel=1e-3, abs=1e-3)
        assert H[s, s].item() == pytest.approx(d2, rel=1e-3, abs=1e-3)
        assert order3[s].item() == pytest.approx(d3, rel=1e-3, abs=1e-4)


def test_offdiagonal_matches_polarization(mlp, batch, mse, param_struct, direction):
    """Hbar[s,t] = (phi''_{v_s+v_t} - phi''_{v_s} - phi''_{v_t}) / 2."""
    S = param_struct.nb_groups
    if S < 2:
        pytest.skip("needs at least two groups")

    x, y = batch(mlp)
    full_loss = mse(mlp)
    H, _, _ = compute_Hg(param_struct, full_loss, x, y, direction)

    for s in range(S):
        for t in range(s + 1, S):
            vs = _masked(direction, param_struct, s)
            vt = _masked(direction, param_struct, t)
            vst = tuple(a + b for a, b in zip(vs, vt))
            _, dss, _ = _derivs_fd(param_struct, full_loss, x, y, vs)
            _, dtt, _ = _derivs_fd(param_struct, full_loss, x, y, vt)
            _, dstst, _ = _derivs_fd(param_struct, full_loss, x, y, vst)
            expected = 0.5 * (dstst - dss - dtt)
            assert H[s, t].item() == pytest.approx(expected, rel=1e-3, abs=1e-5)


# --------------------------------------------------------------------------
# 2. closed-form oracle
# --------------------------------------------------------------------------


def test_quadratic_oracle(f64, device):
    """On L = .5 t^T A t + b^T t, Hbar = V A V^T, gbar = V(A t + b), order3 = 0."""
    sizes = [3, 4, 2]
    model = Quadratic(sizes).to(device=device, dtype=f64)
    ps = ParamStructure([{"params": [p]} for p in model.blocks])
    direction = tuple(torch.randn_like(p) for p in ps.tup_params)

    x = torch.zeros(1, 1, device=device, dtype=f64)
    def full_loss(x_, y_):
        return model(x_).mean()

    H, g, order3 = compute_Hg(ps, full_loss, x, None, direction)

    n = sum(sizes)
    idx = [0] + list(torch.cumsum(torch.tensor(sizes), 0).tolist())
    V = torch.zeros(len(sizes), n, device=device, dtype=f64)
    u = torch.cat([d.reshape(-1) for d in direction])
    for s in range(len(sizes)):
        V[s, idx[s]:idx[s + 1]] = u[idx[s]:idx[s + 1]]

    assert torch.allclose(H, V @ model.A @ V.T, atol=1e-9)
    assert torch.allclose(g, V @ (model.A @ model.theta() + model.b), atol=1e-9)
    assert torch.allclose(order3, torch.zeros_like(order3), atol=1e-10)


# --------------------------------------------------------------------------
# 3. algebraic invariants
# --------------------------------------------------------------------------


def test_H_is_symmetric(mlp, batch, mse, param_struct, direction):
    x, y = batch(mlp)
    H, _, _ = compute_Hg(param_struct, mse(mlp), x, y, direction)
    assert torch.allclose(H, H.T, atol=1e-12)


@pytest.mark.parametrize("c", [-2.0, 0.5, 3.0])
def test_homogeneity_in_the_direction(mlp, batch, mse, param_struct, direction, c):
    """gbar is degree 1, Hbar degree 2, order3 degree 3 in u."""
    x, y = batch(mlp)
    full_loss = mse(mlp)
    H1, g1, o1 = compute_Hg(param_struct, full_loss, x, y, direction)
    scaled = tuple(c * d for d in direction)
    H2, g2, o2 = compute_Hg(param_struct, full_loss, x, y, scaled)

    assert torch.allclose(g2, c * g1, rtol=1e-9, atol=1e-11)
    assert torch.allclose(H2, c**2 * H1, rtol=1e-9, atol=1e-11)
    assert torch.allclose(o2, c**3 * o1, rtol=1e-8, atol=1e-10)


def test_trivial_partition_is_the_total_sum(mlp, batch, mse):
    """Coarsening the partition sums the entries: the S=1 case must equal the
    full contraction of the canonical case."""
    x, y = batch(mlp)
    full_loss = mse(mlp)

    ps_can = ParamStructure(build_partition.canonical(mlp)[0])
    ps_tri = ParamStructure(build_partition.trivial(mlp)[0])
    assert ps_can.tup_params == ps_tri.tup_params  # same order, so one direction
    direction = tuple(torch.randn_like(p) for p in ps_can.tup_params)

    Hc, gc, _ = compute_Hg(ps_can, full_loss, x, y, direction)
    Ht, gt, _ = compute_Hg(ps_tri, full_loss, x, y, direction)

    assert gt.item() == pytest.approx(gc.sum().item(), rel=1e-9)
    assert Ht.item() == pytest.approx(Hc.sum().item(), rel=1e-9)


# --------------------------------------------------------------------------
# 4. ordering guard  (regression test for the direction/tup_params mismatch)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PARTITION_BUILDERS))
def test_direction_is_indexed_in_tup_params_order(uniform_mlp, batch, mse, name):
    """compute_Hg must consume `direction` in ParamStructure.tup_params order.

    Uses a uniform-width MLP, where all hidden weights share a shape, so a
    permuted direction broadcasts silently instead of raising.
    """
    ps = ParamStructure(PARTITION_BUILDERS[name](uniform_mlp)[0])
    x, y = batch(uniform_mlp)
    full_loss = mse(uniform_mlp)

    by_struct = {id(p): torch.randn_like(p) for p in ps.tup_params}
    correct = tuple(by_struct[id(p)] for p in ps.tup_params)
    as_model_order = tuple(by_struct[id(p)] for p in uniform_mlp.parameters())

    is_order_preserved = True
    for t1, t2 in zip(correct, as_model_order):
        if (t1 != t2).int().sum() != 0:
            is_order_preserved = False
            break

    H_ok, g_ok, _ = compute_Hg(ps, full_loss, x, y, correct)

    if is_order_preserved:
        pytest.skip(f"{name} preserves model.parameters() order")

    H_bad, g_bad, _ = compute_Hg(ps, full_loss, x, y, as_model_order)
    # If these agree, the ordering is not actually being respected anywhere.
    assert not torch.allclose(g_ok, g_bad), (
        f"partition {name}: permuting `direction` changed nothing, so the "
        "consumer is not honouring tup_params order"
    )
