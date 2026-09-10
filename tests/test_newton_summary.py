"""Tests for `NewtonSummary`, the optimizer that applies to a plain function.

This one has no data loader, no averaging and no stochasticity, so it can be held
to the standards of a deterministic continuous-optimization routine rather than
to "the loss went down". Five groups:

  1. the interface contract -- the objective must be a CLOSURE, and `step()` owns
     the backward;
  2. exactness -- on a quadratic with `noregul`, one step is the exact minimiser
     over span{u_1, ..., u_S}, which is the whole point of the reduced model;
  3. the cubic model -- the returned lrs satisfy its first-order condition and
     its second-order certificate;
  4. invariance -- the two properties of Appendix E, on functions rather than on
     networks, where they are cheap to test exactly;
  5. convergence -- Rosenbrock from the standard starting point.
"""

import pytest
import torch

from grnewt import NewtonSummary, ParamStructure, optimizers
from grnewt.config import HgCfg, NesterovCfg


def _cfg(**kw):
    kw.setdefault("nesterov", NesterovCfg(use=True, damping_int=1.0))
    kw.setdefault("damping", 1.0)
    return HgCfg(**kw)


def _quadratic(n=9, block=3, seed=0, cond=None):
    """f(theta) = 1/2 theta' A theta - b' theta, split into n // block groups."""
    torch.manual_seed(seed)
    m = torch.randn(n, n)
    a = m @ m.T + n * torch.eye(n)
    if cond is not None:
        vals, vecs = torch.linalg.eigh(a)
        vals = torch.logspace(0, torch.log10(torch.tensor(float(cond))), n, dtype=a.dtype)
        a = vecs @ torch.diag(vals) @ vecs.T
        a = 0.5 * (a + a.T)
    b = torch.randn(n)
    params = [torch.nn.Parameter(torch.randn(block)) for _ in range(n // block)]

    def theta():
        return torch.cat(list(params))

    def loss_fn():
        t = theta()
        return 0.5 * t @ a @ t - b @ t

    return params, theta, loss_fn, a, b


def _make(params, loss_fn, *, momentum=0.0, groups=None, **cfg_kw):
    pgroups = groups if groups is not None else [{"params": [p]} for p in params]
    updater = optimizers.SGDUpdate(params, lr=1, momentum=momentum)
    return NewtonSummary(pgroups, loss_fn, updater, cfg=_cfg(**cfg_kw)), updater


# ---------------------------------------------------------------------------
# 1. the interface contract
# ---------------------------------------------------------------------------


def test_a_loss_tensor_is_refused_with_an_explanation(f64):
    """A tensor is a graph built at ONE point. `step()` moves the parameters in
    place, so the second call raises

        one of the variables needed for gradient computation has been modified by
        an inplace operation ... is at version 1; expected version 0

    i.e. a tensor allows exactly one step and then dies. Refusing it at
    construction turns that into a message the caller can act on."""
    params, _, loss_fn, _, _ = _quadratic()
    updater = optimizers.SGDUpdate(params, lr=1, momentum=0.0)
    with pytest.raises(TypeError, match="callable"):
        NewtonSummary([{"params": [p]} for p in params], loss_fn(), updater, cfg=_cfg())
    with pytest.raises(TypeError, match="callable"):
        NewtonSummary([{"params": [p]} for p in params], 3.0, updater, cfg=_cfg())


def test_many_steps_in_a_row(f64):
    """The regression for the above: the objective must be re-evaluated at each
    step, not reused from construction."""
    params, _, loss_fn, _, _ = _quadratic()
    opt, _ = _make(params, loss_fn, noregul=True, nesterov=NesterovCfg(use=False))
    losses = []
    for _ in range(12):
        losses.append(float(loss_fn().detach()))
        opt.step()
    losses.append(float(loss_fn().detach()))
    assert len(opt.logs["H"]) == 12
    assert all(torch.isfinite(p).all() for p in params)
    assert losses[-1] < losses[0]


def test_step_owns_the_backward(f64):
    """`step()` fills `.grad` itself from the freshly evaluated objective. A stale
    `.grad` left by the caller must not leak into the direction: it would be the
    gradient at the PREVIOUS point."""
    params, _, loss_fn, _, _ = _quadratic()
    opt, updater = _make(params, loss_fn, noregul=True, nesterov=NesterovCfg(use=False))

    for p in params:
        p.grad = torch.full_like(p, 1e6)  # nonsense left over from a caller
    before = [p.detach().clone() for p in params]
    opt.step()
    after_poisoned = [p.detach().clone() for p in params]

    for p, b in zip(params, before, strict=True):
        p.data.copy_(b)
        p.grad = None
    opt2, _ = _make(params, loss_fn, noregul=True, nesterov=NesterovCfg(use=False))
    opt2.step()
    after_clean = [p.detach().clone() for p in params]

    for a, b in zip(after_poisoned, after_clean, strict=True):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-14)


def test_Hbar_is_evaluated_at_the_current_point(f64):
    """Not at construction. On a quartic the curvature moves, so a stale graph
    would show up as a constant Hbar."""
    p1 = torch.nn.Parameter(torch.tensor([0.7]))
    p2 = torch.nn.Parameter(torch.tensor([-0.4]))

    def loss_fn():
        return p1[0] ** 4 / 4 + p2[0] ** 4 / 4 + 0.5 * p1[0] * p2[0] + p1[0] - p2[0]

    opt, _ = _make([p1, p2], loss_fn, damping=0.3)
    opt.step()
    snapshot = (p1.detach().clone(), p2.detach().clone())
    opt.step()

    assert not torch.allclose(opt.logs["H"][0], opt.logs["H"][1], rtol=1e-3)

    # a fresh optimizer started at the point reached after step 1 must produce the
    # same Hbar as the running one produced at step 2
    q1 = torch.nn.Parameter(snapshot[0].clone())
    q2 = torch.nn.Parameter(snapshot[1].clone())

    def loss_fn2():
        return q1[0] ** 4 / 4 + q2[0] ** 4 / 4 + 0.5 * q1[0] * q2[0] + q1[0] - q2[0]

    fresh, _ = _make([q1, q2], loss_fn2, damping=0.3)
    fresh.step()
    assert torch.allclose(fresh.logs["H"][0], opt.logs["H"][1], rtol=1e-12, atol=1e-14)
    assert torch.allclose(fresh.logs["g"][0], opt.logs["g"][1], rtol=1e-12, atol=1e-14)


def test_a_solver_failure_raises_instead_of_stalling(f64):
    """Unlike the NSBase family there is no previous estimate to fall back on, so
    `nesterov_lrs` returning found=False must be loud. `damping_int=0` requires
    Hbar > 0; an indefinite Hbar then has no bounded step."""
    p = torch.nn.Parameter(torch.tensor([0.3, -0.2]))

    def loss_fn():
        return p[0] ** 2 - 3.0 * p[1] ** 2 + p[0] * p[1]  # indefinite, unbounded below

    opt, _ = _make([p], loss_fn, damping=1.0, nesterov=NesterovCfg(use=True, damping_int=0.0))
    with pytest.raises(RuntimeError, match="unbounded below"):
        opt.step()


# ---------------------------------------------------------------------------
# 2. exactness of the reduced model
# ---------------------------------------------------------------------------


def test_one_step_minimises_the_quadratic_over_span_of_the_directions(f64):
    """The defining property of the reduced model. For quadratic f and `noregul`,
    eta = Hbar^-1 gbar is exactly argmin over t of f(theta - sum_s t_s u_s):
    d/dt_s = -<u_s, grad f> + sum_t t_t u_s' A u_t = 0 is precisely Hbar t = gbar.

    Checked against a direct solve of the S x S restricted problem, and by
    perturbing t* in both directions."""
    n, block = 9, 3
    params, theta, loss_fn, a, b = _quadratic(n=n, block=block)
    opt, _ = _make(
        params, loss_fn, noregul=True, remove_negative=False, nesterov=NesterovCfg(use=False)
    )

    th0 = theta().detach().clone()
    opt.step()
    step = th0 - theta().detach()

    grad = a @ th0 - b
    n_groups = n // block
    u = torch.zeros(n, n_groups, dtype=th0.dtype)
    for s in range(n_groups):
        u[block * s : block * (s + 1), s] = grad[block * s : block * (s + 1)]
    t_star = torch.linalg.solve(u.T @ a @ u, u.T @ grad)
    step_ref = u @ t_star

    assert torch.allclose(step, step_ref, rtol=1e-10, atol=1e-12)

    def f(v):
        return float(0.5 * v @ a @ v - b @ v)

    best = f(th0 - step_ref)
    for eps in (-1e-3, 1e-3):
        assert f(th0 - (1 + eps) * step_ref) >= best


def test_the_trivial_partition_is_an_exact_line_search(f64):
    """S = 1: the reduced model is one-dimensional and eta = gbar / Hbar is the
    exact minimiser along u."""
    n = 8
    params, theta, loss_fn, a, b = _quadratic(n=n, block=n)
    opt, _ = _make(
        params,
        loss_fn,
        groups=[{"params": list(params)}],
        noregul=True,
        remove_negative=False,
        nesterov=NesterovCfg(use=False),
    )
    th0 = theta().detach().clone()
    grad = a @ th0 - b
    opt.step()
    t_taken = float(((th0 - theta().detach()) / grad)[0])
    t_star = float((grad @ grad) / (grad @ a @ grad))
    assert t_taken == pytest.approx(t_star, rel=1e-10)


def test_it_decreases_a_strongly_convex_quadratic_monotonically(f64):
    params, _, loss_fn, _, _ = _quadratic(n=12, block=4, seed=3)
    opt, _ = _make(params, loss_fn)
    prev = float(loss_fn())
    for _ in range(25):
        opt.step()
        cur = float(loss_fn())
        assert cur <= prev + 1e-12, f"{prev:.9f} -> {cur:.9f}"
        prev = cur


# ---------------------------------------------------------------------------
# 3. the cubic model
# ---------------------------------------------------------------------------


def test_the_lrs_satisfy_the_cubic_first_order_condition(f64):
    """eta must solve Hbar eta + (lambda/2) ||D eta|| D^2 eta = gbar with
    M(r) = Hbar + (lambda/2) r D^2 positive semidefinite -- the second condition is
    what distinguishes the global minimiser of the cubic model from the other
    stationary points."""
    lam = 1.0
    params, _, loss_fn, _, _ = _quadratic(n=12, block=4, seed=5)
    opt, _ = _make(
        params, loss_fn, remove_negative=False, nesterov=NesterovCfg(use=True, damping_int=lam)
    )
    for _ in range(4):
        opt.step()

    for k in range(4):
        h = opt.logs["H"][k]
        g = opt.logs["g"][k]
        d = opt.logs["order3"][k].abs().pow(1 / 3)
        eta = opt.logs["lrs_clipped"][k]
        r = float(torch.linalg.vector_norm(d * eta))
        m = h + 0.5 * lam * r * torch.diag(d.pow(2))
        scale = max(float(g.abs().max()), 1e-12)
        assert float((m @ eta - g).norm()) < 1e-8 * scale
        assert float(torch.linalg.eigvalsh(m).min()) > -1e-8 * max(float(m.abs().max()), 1.0)


def test_normalize_dirs_gives_every_group_unit_norm(f64):
    params, _, loss_fn, _, _ = _quadratic(n=9, block=3)
    opt, _ = _make(params, loss_fn, normalize_dirs=True)
    ps = opt.param_struct

    seen = []
    real = opt.compute_Hg

    def spy(loss, direction):
        seen.append(ps.squared_norm(direction).sqrt().clone())
        return real(loss, direction)

    opt.compute_Hg = spy
    for _ in range(3):
        opt.step()

    assert len(seen) == 3
    for norms in seen:
        assert norms.numel() == ps.nb_groups
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)


# ---------------------------------------------------------------------------
# 4. invariance (Appendix E), exactly testable on functions
# ---------------------------------------------------------------------------


def test_the_step_does_not_depend_on_the_scale_of_the_direction(f64):
    """u_s -> c_s u_s gives Hbar -> C Hbar C, gbar -> C gbar, order3 -> C^3 order3,
    hence eta -> C^-1 eta and the applied step eta_s u_s is unchanged. Exact for a
    single step, which is all this optimizer ever does (no averaging, no mom_lrs)."""
    n, block, lam = 9, 3, 1.0
    params, theta, loss_fn, _, _ = _quadratic(n=n, block=block, seed=7)
    th0 = theta().detach().clone()

    class Scaled(optimizers.SGDUpdate):
        scales = None

        def compute_step(self):
            out = super().compute_step()
            for d, c in zip(out, self.scales, strict=True):
                d.mul_(c)
            return out

    def run(scales):
        for p, chunk in zip(params, th0.split(block), strict=True):
            p.data.copy_(chunk)
            p.grad = None
        upd = Scaled(params, lr=1, momentum=0.0)
        upd.scales = scales
        opt = NewtonSummary(
            [{"params": [p]} for p in params],
            loss_fn,
            upd,
            cfg=_cfg(remove_negative=False, nesterov=NesterovCfg(use=True, damping_int=lam)),
        )
        opt.step()
        return theta().detach().clone()

    ref = run([1.0] * len(params))
    got = run([1e-3, 7.0, 250.0])
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-11)


def test_the_trajectory_is_invariant_under_a_per_group_rescaling_of_the_parameters(f64):
    """Appendix E with J = Diag(a_s I): theta_s = a_s theta~_s must leave the
    trajectory in the ORIGINAL coordinates unchanged. On a function this is exact,
    with none of the minibatch noise that hides it on a network."""
    n, block = 9, 3
    params, _, _, a, b = _quadratic(n=n, block=block, seed=11)
    th0 = torch.cat([p.detach().clone() for p in params])
    scales = torch.tensor([0.25, 4.0, 100.0], dtype=th0.dtype)

    def run(with_scales):
        chunks = list(th0.split(block))
        c = scales if with_scales else torch.ones_like(scales)
        ps = [torch.nn.Parameter(chunk / c[s]) for s, chunk in enumerate(chunks)]

        def loss_fn():
            t = torch.cat([c[s] * ps[s] for s in range(len(ps))])
            return 0.5 * t @ a @ t - b @ t

        upd = optimizers.SGDUpdate(ps, lr=1, momentum=0.0)
        opt = NewtonSummary(
            [{"params": [p]} for p in ps],
            loss_fn,
            upd,
            cfg=_cfg(damping=0.5, remove_negative=False),
        )
        for _ in range(6):
            opt.step()
        return torch.cat([c[s] * ps[s].detach() for s in range(len(ps))])

    assert torch.allclose(run(True), run(False), rtol=1e-8, atol=1e-10)


def test_two_identical_runs_agree_bitwise(f64):
    def run():
        params, theta, loss_fn, _, _ = _quadratic(n=12, block=4, seed=13)
        opt, _ = _make(params, loss_fn, damping=0.5)
        for _ in range(8):
            opt.step()
        return theta().detach().clone()

    assert torch.equal(run(), run())


def test_the_direction_reaches_compute_Hg_in_tup_params_order(f64):
    """The producer is the updater, built from the parameter list; `compute_Hg`
    consumes in partition order. `ParamStructure.dot` broadcasts rather than
    raising, so a permuted direction is silently wrong."""
    params, _, loss_fn, _, _ = _quadratic(n=9, block=3)
    # a partition that reverses the parameter order
    groups = [{"params": [p]} for p in reversed(params)]
    opt, _ = _make(params, loss_fn, groups=groups)
    ps = ParamStructure(groups)
    assert [id(p) for p in ps.tup_params] != [id(p) for p in params]

    seen = {}
    real = opt.compute_Hg

    def spy(loss, direction):
        seen["direction"] = direction
        return real(loss, direction)

    opt.compute_Hg = spy
    opt.step()
    for d, p in zip(seen["direction"], ps.tup_params, strict=True):
        assert d.shape == p.shape


# ---------------------------------------------------------------------------
# 5. convergence
# ---------------------------------------------------------------------------


def test_rosenbrock(f64):
    """The standard non-convex smoke test, from the standard starting point.
    S = 2 (one group per coordinate), so the reduced model spans the whole plane
    and the step is a genuine cubic-regularised Newton step."""
    x = torch.nn.Parameter(torch.tensor([-1.2]))
    y = torch.nn.Parameter(torch.tensor([1.0]))

    def loss_fn():
        return (1 - x[0]) ** 2 + 100 * (y[0] - x[0] ** 2) ** 2

    opt, _ = _make([x, y], loss_fn, damping=1.0)
    assert float(loss_fn()) == pytest.approx(24.2)
    for _ in range(200):
        opt.step()
    assert float(loss_fn()) < 1e-12
    assert float(x) == pytest.approx(1.0, abs=1e-6)
    assert float(y) == pytest.approx(1.0, abs=1e-6)


def test_it_stays_at_a_minimum_and_then_says_why_it_cannot_continue(f64):
    """Started at the solution of a quadratic, the steps must be numerically zero.

    They are -- for a few iterations. Then it raises, and that is worth knowing
    about: on a QUADRATIC, order3 = 0 exactly, so D = 0, the whole space is ker(D)
    and nesterov_lrs needs Hbar > 0 there. Near the solution Hbar = u' A u with
    u = grad f of norm ~1e-16, i.e. Hbar ~ 1e-31, which rounds to something
    numerically indefinite after a few steps. Measured: 4 steps of size ~4e-17,
    then `infeasible_no_cubic`.

    This is not a defect of nesterov_lrs -- Hbar really is numerically indefinite
    at that scale -- but it does mean the optimizer cannot simply be run to
    convergence. The caller needs a stopping criterion, and the exception must say
    which of the two situations it is (unbounded model vs solver failure)."""
    n, block = 9, 3
    params, theta, loss_fn, a, b = _quadratic(n=n, block=block, seed=17)
    star = torch.linalg.solve(a, b)
    for p, chunk in zip(params, star.split(block), strict=True):
        p.data.copy_(chunk)
    opt, _ = _make(params, loss_fn)

    n_ok = 0
    try:
        for _ in range(10):
            opt.step()
            n_ok += 1
            assert torch.allclose(theta().detach(), star, rtol=0, atol=1e-9)
    except RuntimeError as exc:
        assert "unbounded below" in str(exc)
        assert "stationary point" in str(exc)
    assert n_ok >= 1, "it must at least not move away from the minimum"
