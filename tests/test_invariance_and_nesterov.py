"""
Two checks:

  1. the affine layer-wise invariance property (Appendix E);
  2. nesterov_lrs as a solver.

Under the reparameterization theta_s = a_s * theta~_s the exact scaling laws are

    u~_s      = a_s   u_s          (gradient direction transforms contravariantly)
    g~_s      = a_s^2 g_s
    H~_{s,t}  = a_s^2 a_t^2 H_{s,t}
    order3~_s = a_s^6 order3_s
    D~_s      = a_s^2 D_s          (D = |order3|^(1/3))
    eta~_s    = eta_s / a_s^2      (both with and without cubic regularization)

so that the induced step on theta is unchanged. Testing the chain of scaling
laws is far more robust than comparing two training trajectories.
"""

import pytest
import torch

from grnewt import ParamStructure, compute_Hg, nesterov_lrs
from grnewt import partition as build_partition

# --------------------------------------------------------------------------
# reparameterized wrapper
# --------------------------------------------------------------------------


class Reparam(torch.nn.Module):
    """Same function as `base`, but with parameters theta~_s = theta_s / a_s."""

    def __init__(self, base, scales):
        super().__init__()
        self._base = [base]  # in a list: keeps base's params out of the registry
        self.names = [n for n, _ in base.named_parameters()]
        self.scales = list(scales)
        self.tilde = torch.nn.ParameterList(
            torch.nn.Parameter(p.detach().clone() / a)
            for (_, p), a in zip(base.named_parameters(), self.scales)
        )

    def forward(self, x):
        params = {n: a * p for n, a, p in zip(self.names, self.scales, self.tilde)}
        return torch.func.functional_call(self._base[0], params, (x,))


@pytest.fixture
def scales():
    return [0.5, 2.0, 4.0, 0.25, 1.5, 3.0, 0.75, 1.25]


# --------------------------------------------------------------------------
# invariance
# --------------------------------------------------------------------------


def test_reparam_preserves_the_function(mlp, batch, scales):
    x, _ = batch(mlp)
    tilde = Reparam(mlp, scales[: len(list(mlp.parameters()))])
    assert torch.allclose(mlp(x), tilde(x), atol=1e-10)


def test_Hg_scaling_laws(mlp, batch, mse, scales):
    x, y = batch(mlp)
    a = torch.tensor(scales[: len(list(mlp.parameters()))], dtype=x.dtype, device=x.device)

    ps = ParamStructure(build_partition.canonical(mlp)[0])
    u = tuple(torch.randn_like(p) for p in ps.tup_params)
    H, g, o3 = compute_Hg(ps, mse(mlp), x, y, u)

    tilde = Reparam(mlp, a.tolist()).to(x.device)
    ps_t = ParamStructure(build_partition.canonical(tilde)[0])
    u_t = tuple(ai * ui for ai, ui in zip(a, u))
    H_t, g_t, o3_t = compute_Hg(ps_t, mse(tilde), x, y, u_t)

    A = a.pow(2)
    assert torch.allclose(g_t, A * g, rtol=1e-8, atol=1e-11)
    assert torch.allclose(H_t, A[:, None] * H * A[None, :], rtol=1e-8, atol=1e-11)
    assert torch.allclose(o3_t, a.pow(6) * o3, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("damping_int", [0.0, 1.0, 10.0])
def test_lrs_are_invariant(mlp, batch, mse, scales, damping_int):
    """eta~ = eta / a^2, i.e. the step on theta is identical. This is the
    property Appendix E claims; it must survive the cubic regularization."""
    x, y = batch(mlp)
    a = torch.tensor(scales[: len(list(mlp.parameters()))], dtype=x.dtype, device=x.device)

    ps = ParamStructure(build_partition.canonical(mlp)[0])
    u = tuple(torch.randn_like(p) for p in ps.tup_params)
    H, g, o3 = compute_Hg(ps, mse(mlp), x, y, u)

    A = a.pow(2)
    H_t, g_t, o3_t = A[:, None] * H * A[None, :], A * g, a.pow(6) * o3

    if damping_int == 0.0:
        eta = torch.linalg.solve(H, g)
        eta_t = torch.linalg.solve(H_t, g_t)
    else:
        eta, log = nesterov_lrs(H, g, o3.abs().pow(1 / 3), damping_int=damping_int)
        eta_t, log_t = nesterov_lrs(H_t, g_t, o3_t.abs().pow(1 / 3), damping_int=damping_int)
        if not (log["found"] and log_t["found"]):
            pytest.skip("root-finder did not converge on this instance")
        # r = ||D eta|| is itself invariant
        assert log_t["r"].item() == pytest.approx(log["r"].item(), rel=1e-6)

    assert torch.allclose(eta_t, eta / A, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# nesterov_lrs as a solver
# --------------------------------------------------------------------------


def _residual(H, g, D_vec, lrs, damping_int):
    """||lrs - (H + .5 * lambda * ||D lrs|| D^2)^{-1} g||, the defining equation."""
    D = D_vec.diag().to(torch.float64)
    H, g, lrs = H.double(), g.double(), lrs.double()
    r = (D @ lrs).norm()
    M = H + 0.5 * damping_int * r * (D @ D)
    return (lrs - torch.linalg.solve(M, g)).norm().item()


def _spd(S, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(S, S, generator=g, dtype=torch.float64)
    return M @ M.T + S * torch.eye(S, dtype=torch.float64)


@pytest.mark.parametrize("S", [1, 3, 8])
@pytest.mark.parametrize("damping_int", [0.5, 5.0])
def test_solver_satisfies_its_own_equation_H_pd(S, damping_int):
    H = _spd(S)
    g = torch.randn(S, dtype=torch.float64)
    D = torch.rand(S, dtype=torch.float64) + 0.5

    lrs, log = nesterov_lrs(H, g, D, damping_int=damping_int)
    assert log["found"], log
    assert _residual(H, g, D, lrs, damping_int) < 1e-6


@pytest.mark.parametrize("damping_int", [1.0, 20.0])
def test_solver_handles_indefinite_H(damping_int):
    S = 5
    H = _spd(S)
    H = H - 2 * torch.linalg.eigvalsh(H).max() * torch.eye(S, dtype=torch.float64)
    assert torch.linalg.eigvalsh(H).min() < 0
    g = torch.randn(S, dtype=torch.float64)
    D = torch.rand(S, dtype=torch.float64) + 0.5

    lrs, log = nesterov_lrs(H, g, D, damping_int=damping_int)
    if not log["found"]:
        pytest.skip("no root for this instance; documented failure mode")
    assert _residual(H, g, D, lrs, damping_int) < 1e-5


@pytest.mark.xfail(
    reason="D_inv = (1/order3_).diag() is built before the singularity check, "
    "and threshold_D_sing is compared against |order3|^(1/3), so it only "
    "fires below ~1e-15. A vanishing third derivative (e.g. a final linear "
    "layer under MSE) produces inf/nan instead of taking the Numerical branch.",
    strict=False,
)
@pytest.mark.parametrize("zero_at", [0, 3])
def test_solver_survives_a_vanishing_third_derivative(zero_at):
    S = 5
    H = _spd(S)
    H = H - 2 * torch.linalg.eigvalsh(H).max() * torch.eye(S, dtype=torch.float64)
    g = torch.randn(S, dtype=torch.float64)
    D = torch.rand(S, dtype=torch.float64) + 0.5
    D[zero_at] = 0.0

    lrs, log = nesterov_lrs(H, g, D, damping_int=1.0)
    assert (lrs is None) or torch.isfinite(lrs).all()


def test_solver_scale_invariance():
    """nesterov_lrs(A H A, A g, a^2 D) == nesterov_lrs(H, g, D) / A, A = a^2."""
    S = 6
    H, g = _spd(S), torch.randn(S, dtype=torch.float64)
    D = torch.rand(S, dtype=torch.float64) + 0.5
    a = torch.tensor([0.5, 2.0, 1.0, 4.0, 0.25, 1.5], dtype=torch.float64)
    A = a.pow(2)

    lrs, log = nesterov_lrs(H, g, D, damping_int=3.0)
    lrs_t, log_t = nesterov_lrs(A[:, None] * H * A[None, :], A * g, A * D, damping_int=3.0)
    assert log["found"] and log_t["found"]
    assert torch.allclose(lrs_t, lrs / A, rtol=1e-6, atol=1e-9)


@pytest.mark.timeout(30)
def test_solver_terminates_on_a_nasty_instance():
    """Guards the unbounded `while f(x1) >= 0: x1 *= 3` bracketing loop."""
    # S = 4
    H = torch.diag(torch.tensor([1e8, 1.0, -1e-6, 1e-8], dtype=torch.float64))
    g = torch.tensor([1e6, 0.0, -1e-8, 1e3], dtype=torch.float64)
    D = torch.tensor([1e-4, 1e4, 1.0, 1e-3], dtype=torch.float64)
    lrs, log = nesterov_lrs(H, g, D, damping_int=1.0)
    assert (lrs is None) or torch.isfinite(lrs).all()
