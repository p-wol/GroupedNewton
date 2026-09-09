"""Regression tests for the defects found in commit 3c87171f (2026-08-21).

Each test names the defect it guards and fails on the unfixed code. The last
two encode the two facts about the invariance group that Appendix E of
arXiv:2312.03885 does not state.
"""

import types

import pytest
import torch

from grnewt import (
    NewtonSummaryMovexpAvg,
    ParamStructure,
    compute_Hg,
    compute_Hg_batched,
    nesterov_lrs,
    optimizers,
)
from grnewt import partition as build_partition
from grnewt.config import HgCfg
from grnewt.nesterov import compute_x0
from grnewt.newton_summary_uniform_avg import NewtonSummaryUniformAvg

# ==========================================================================
# N1 -- the uniform average aliased its own input
# ==========================================================================


def _avg_harness(period, period_hg=1):
    o = types.SimpleNamespace()
    o.cfg = types.SimpleNamespace(
        period_hg=period_hg,
        uniform_avg=types.SimpleNamespace(period=period, warmup=0),
    )
    o.dct_HgD_avgs = {k: None for k in ["H_use", "H_up", "g_use", "g_up", "D_use", "D_up"]}
    o.update_uniform_avg = types.MethodType(NewtonSummaryUniformAvg.update_uniform_avg, o)
    return o


@pytest.mark.parametrize("period", [1, 2, 3, 5])
def test_uniform_avg_matches_the_reference_average(period):
    """`_use` must be the running mean of every sample seen since the start of
    the previous period. Before the fix it was exactly 0 at step 0 and dropped
    the first sample of every period thereafter."""
    o = _avg_harness(period)
    n_steps = 4 * period
    samples = [torch.tensor([float(k + 1)], dtype=torch.float64) for k in range(n_steps)]

    for k in range(n_steps):
        o.step_counter = k
        _, g_use, _ = o.update_uniform_avg(
            samples[k].clone(), samples[k].clone(), samples[k].clone()
        )

        # reference: mean over the previous complete period plus the current one
        blk = k // period
        first = max(0, (blk - 1) * period)
        window = samples[first : k + 1]
        expected = sum(float(s) for s in window) / len(window)
        assert float(g_use) == pytest.approx(expected, rel=1e-12), (
            f"period={period}, step={k}: got {float(g_use)}, expected {expected}"
        )


@pytest.mark.parametrize("period", [1, 3])
def test_uniform_avg_accumulators_never_alias(period):
    """`_use`, `_up` and the incoming sample must be three distinct storages."""
    o = _avg_harness(period)
    for k in range(3 * period):
        o.step_counter = k
        curr = torch.ones(2, dtype=torch.float64) * (k + 1)
        o.update_uniform_avg(curr.clone(), curr, curr.clone())
        ptrs = {
            "use": o.dct_HgD_avgs["g_use"].data_ptr(),
            "up": o.dct_HgD_avgs["g_up"].data_ptr(),
            "curr": curr.data_ptr(),
        }
        assert len(set(ptrs.values())) == 3, f"step {k}: aliased storages {ptrs}"


# ==========================================================================
# N2 -- `direction` produced in model.parameters() order, consumed in
#       tup_params order
# ==========================================================================


def _uniform_mlp(width=6, depth=3):
    layers = []
    for i in range(depth):
        layers.append(torch.nn.Linear(width, width))
        if i < depth - 1:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


PARTITIONS = {
    "canonical": build_partition.canonical,
    "trivial": build_partition.trivial,
    "wb": build_partition.wb,
    "blocks-2": lambda m: build_partition.blocks(m, 2),
}


@pytest.mark.parametrize("name", sorted(PARTITIONS))
def test_direction_reaches_compute_Hg_in_tup_params_order(f64, name, monkeypatch):
    """The seam between the updater and ParamStructure.

    `ParamStructure.dot` contracts with `(p1 * p2).sum()`, which BROADCASTS, so
    a permuted `direction` produces a silently wrong Hbar instead of raising.
    """
    import grnewt.newton_summary_movexp_avg as ns

    model = _uniform_mlp()
    pgroups, _ = PARTITIONS[name](model)
    ps = ParamStructure(pgroups)

    seen = {}
    orig = ns.compute_Hg

    def spy(param_struct, loss, direction, **kw):
        seen["direction"] = direction
        return orig(param_struct, loss, direction, **kw)

    monkeypatch.setattr(ns, "compute_Hg", spy)

    X = torch.randn(16, 6)
    Y = torch.randn(16, 6)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=8)

    def full_loss(a, b):
        return ((model(a) - b) ** 2).mean()

    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    opt = NewtonSummaryMovexpAvg(
        pgroups,
        full_loss,
        loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=HgCfg(),
    )
    x, y = next(iter(loader))
    updater.zero_grad()
    full_loss(x, y).backward()
    opt.step()

    direction = seen["direction"]
    assert len(direction) == len(ps.tup_params)
    for d, p in zip(direction, ps.tup_params, strict=False):
        assert d.shape == p.shape, (
            f"partition {name}: direction entry {tuple(d.shape)} paired with "
            f"parameter {tuple(p.shape)}"
        )
    by_id = {id(p): updater.state[p]["update"] for p in model.parameters()}
    for d, p in zip(direction, ps.tup_params, strict=False):
        assert torch.equal(d, by_id[id(p)])


# ==========================================================================
# N3 -- the hard-case interior bracket used the wrong shift
# ==========================================================================


def test_hard_case_interior_is_solved_not_refused(f64):
    """kappa_min < 0 simple, next eigenvalue > 0, b orthogonal to the kappa_min
    eigenvector. Before the fix this returned ("hard_case_bracket_failed", None)
    although a unique PSD-certified solution exists."""
    kappa = torch.tensor([-1.0, 2.0, 3.0], dtype=torch.float64)
    H = torch.diag(kappa)
    D = torch.ones(3, dtype=torch.float64)
    g = torch.tensor([0.0, 20.0, 20.0], dtype=torch.float64)
    lam = 1.0

    lrs, log = nesterov_lrs(H, g, D, damping_int=lam)
    assert log["found"], log
    assert log["hard_case"]

    r = float(torch.linalg.vector_norm(D * lrs))
    M = H + 0.5 * lam * r * torch.diag(D.pow(2))
    assert torch.linalg.eigvalsh(M).min() >= -1e-10, "M(r_*) must be PSD"
    assert float((M @ lrs - g).norm()) < 1e-9
    assert r == pytest.approx(5.474860754766346, rel=1e-9)


# ==========================================================================
# N4 / N5 -- the two compute_Hg implementations must agree everywhere
# ==========================================================================


@pytest.mark.parametrize("noregul", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, -1])
def test_hg_reference_and_batched_agree(f64, noregul, diagonal, chunk_size):
    model = _uniform_mlp(width=5, depth=4)
    ps = ParamStructure(build_partition.canonical(model)[0])
    x, y = torch.randn(12, 5), torch.randn(12, 5)

    def full_loss(a, b):
        return ((model(a) - b) ** 2).mean()

    u = tuple(torch.randn_like(p) for p in ps.tup_params)
    kw = dict(noregul=noregul, diagonal=diagonal)

    loss = full_loss(x, y)
    H0, g0, o0 = compute_Hg(ps, loss, u, **kw)
    H1, g1, o1 = compute_Hg_batched(ps, loss, u, chunk_size=chunk_size, **kw)

    assert torch.allclose(g0, g1, rtol=1e-9, atol=1e-12)
    assert torch.allclose(H0, H1, rtol=1e-8, atol=1e-11)
    assert torch.allclose(o0, o1, rtol=1e-7, atol=1e-10)

    if diagonal:
        # the entries actually written must be Hbar_ss, not Hbar_s0
        H_full, _, _ = compute_Hg(ps, loss, u, noregul=noregul, diagonal=False)
        assert torch.allclose(H1.diagonal(), H_full.diagonal(), rtol=1e-8, atol=1e-11)


# ==========================================================================
# nesterov_lrs: `found=False` has exactly one legitimate cause
# ==========================================================================


@pytest.mark.parametrize("seed", range(200))
def test_nesterov_fails_only_when_the_model_is_unbounded_below(f64, seed):
    """Prop. 2: inf T = -inf iff Hbar restricted to ker(D) is not PD. Any other
    `found=False` is a solver defect, not a property of the instance.

    NB the exponents below are Python floats on purpose: `10 ** torch.randint(...)`
    is integer arithmetic and silently returns 0 for negative exponents.
    """
    gen = torch.Generator().manual_seed(seed)
    S = int(torch.randint(2, 9, (1,), generator=gen))
    A = torch.randn(S, S, generator=gen, dtype=torch.float64)
    H = A + A.T
    H = H * 10.0 ** int(torch.randint(-4, 5, (1,), generator=gen))
    g = torch.randn(S, generator=gen, dtype=torch.float64)
    d = torch.rand(S, generator=gen, dtype=torch.float64).pow(3)
    # force exact zeros in D (order3_s = 0 whenever the loss is quadratic in s)
    n_zero = int(torch.randint(0, max(1, S // 2) + 1, (1,), generator=gen))
    if n_zero:
        d[torch.randperm(S, generator=gen)[:n_zero]] = 0.0
    # and force an exact zero in g (a frozen group), which is what triggers the
    # hard case in the diagonal regime
    if int(torch.randint(0, 2, (1,), generator=gen)):
        g[int(torch.randint(0, S, (1,), generator=gen))] = 0.0

    lam = 10.0 ** int(torch.randint(-2, 3, (1,), generator=gen))
    lrs, log = nesterov_lrs(H, g, d, damping_int=lam)

    Z = (d == 0).nonzero(as_tuple=True)[0]
    if Z.numel() == S:
        # no cubic term anywhere: the model is a bare quadratic, bounded below
        # iff H itself is PD
        bounded = bool(torch.linalg.eigvalsh(H).min() > 0)
    elif Z.numel():
        bounded = bool(torch.linalg.eigvalsh(H[Z][:, Z]).min() > 0)
    else:
        bounded = True

    if not log["found"]:
        assert not bounded, (
            f"seed={seed}: solver refused an instance whose cubic model is "
            f"bounded below (computation={log['x0.computation']})"
        )
        return

    assert torch.isfinite(lrs).all()
    r = float(torch.linalg.vector_norm(d * lrs))
    M = H + 0.5 * lam * r * torch.diag(d.pow(2))

    # The certificate. r and x0 are BOTH exactly affine-invariant (Appendix E,
    # STATE.md V2/D3), so `r >= x0` is an invariant acceptance test -- unlike any
    # absolute threshold on d. A global minimiser of T requires M(r_*) >= 0,
    # i.e. r_* >= x0. compute_x0 reaches ~1e-8 relative accuracy where the
    # secular root does not, so this catches the near-hard failures.
    x0, _ = compute_x0(H, d, damping_int=lam)
    assert x0 is not None
    if x0 > 0:
        assert (r - x0) / x0 > -1e-8, (
            f"seed={seed}: r = {r:.12e} < x0 = {x0:.12e} (violation "
            f"{(x0 - r) / x0:.3e}); M(r) is indefinite so lrs is not a minimiser"
        )

    scale = max(float(H.abs().max()), 1.0) * max(float(lrs.abs().max()), 1.0)
    assert float((M @ lrs - g).norm()) < 1e-6 * max(scale, float(g.abs().max()), 1.0)
    assert torch.linalg.eigvalsh(M).min() > -1e-8 * scale


def _fuzz_instance(seed):
    """The exact generator used by the fuzz above, so a named instance and its
    fuzz counterpart can never drift apart."""
    gen = torch.Generator().manual_seed(seed)
    S = int(torch.randint(2, 9, (1,), generator=gen))
    A = torch.randn(S, S, generator=gen, dtype=torch.float64)
    H = A + A.T
    H = H * 10.0 ** int(torch.randint(-4, 5, (1,), generator=gen))
    g = torch.randn(S, generator=gen, dtype=torch.float64)
    d = torch.rand(S, generator=gen, dtype=torch.float64).pow(3)
    n_zero = int(torch.randint(0, max(1, S // 2) + 1, (1,), generator=gen))
    if n_zero:
        d[torch.randperm(S, generator=gen)[:n_zero]] = 0.0
    if int(torch.randint(0, 2, (1,), generator=gen)):
        g[int(torch.randint(0, S, (1,), generator=gen))] = 0.0
    lam = 10.0 ** int(torch.randint(-2, 3, (1,), generator=gen))
    return H, g, d, lam


def test_near_hard_case_resolves_kappa_min(f64):
    """The instance that motivated `_eigh_invariant`.

    cond(D_R)^2 = 8.5e16, and the root sits at relative distance 1.3e-13 from
    x0. Plain `eigh` returns kappa_min with 9.7e-4 relative error -- legal, since
    it only guarantees an ABSOLUTE eps*||K|| -- and the solver then returned
    r = 68527.18 against the reference 68593.5450086, i.e. r < x0 with
    lambda_min(M(r)) = -0.27: not a minimiser.

    Reference values below come from an 80-digit mpmath solution of the same
    scalar equation.
    """
    H, g, d, lam = _fuzz_instance(80)
    assert float(d[d > 0].max() / d[d > 0].min()) ** 2 > 1e16  # the hard regime

    lrs, log = nesterov_lrs(H, g, d, damping_int=lam)
    assert log["found"], log
    assert log["n_deflations"] >= 1, "this instance must trigger the deflation"

    r = float(torch.linalg.vector_norm(d * lrs))
    x0, _ = compute_x0(H, d, damping_int=lam)
    assert r >= x0 * (1 - 1e-12), f"r = {r:.12e} < x0 = {x0:.12e}"
    assert r == pytest.approx(68593.5450086275839, rel=1e-11)

    eta_ref = torch.tensor(
        [7311564.596, -616.7255204, -3019.081505, 14513.70496, 2172960.37],
        dtype=torch.float64,
    )
    assert torch.allclose(lrs, eta_ref, rtol=1e-9)

    M = H + 0.5 * lam * r * torch.diag(d.pow(2))
    assert torch.linalg.eigvalsh(M).min() > -1e-9 * float(M.abs().max())


def test_deflation_criterion_is_affine_invariant(f64):
    """`_eigh_invariant` decides on the spectrum of K = D_R^-1 S D_R^-1, which is
    pointwise invariant under the group of Appendix E. So the number of passes,
    the eigenvalues, and the resulting step are all invariant -- which is exactly
    what `threshold_D_sing` (a threshold on d) fails to be."""
    from grnewt.nesterov import _eigh_invariant

    H, g, d, lam = _fuzz_instance(80)
    a = torch.tensor([0.5, 2.0, 4.0, 0.25, 1.5], dtype=torch.float64)
    A = a.pow(2)
    H_t, g_t, d_t = A[:, None] * H * A[None, :], A * g, A * d

    def spectrum(H_, d_):
        K = H_ / (d_.unsqueeze(1) * d_.unsqueeze(0))
        return _eigh_invariant(0.5 * (K + K.T))

    k0, _, n0 = spectrum(H, d)
    k1, _, n1 = spectrum(H_t, d_t)
    assert n0 == n1 >= 1
    assert torch.allclose(k1, k0, rtol=1e-12)

    lrs, log = nesterov_lrs(H, g, d, damping_int=lam)
    lrs_t, log_t = nesterov_lrs(H_t, g_t, d_t, damping_int=lam)
    assert log["found"] and log_t["found"]
    assert log["n_deflations"] == log_t["n_deflations"]
    assert torch.allclose(lrs_t * A, lrs, rtol=1e-8)


# ==========================================================================
# C3 -- the exact invariance group, as opposed to the one Appendix E states
# ==========================================================================


def _summaries(base, names, params, groups, u, x, y):
    pgroups = [{"params": [params[i] for i in gr]} for gr in groups]
    ps = ParamStructure(pgroups)

    def full_loss(a, b):
        out = torch.func.functional_call(base, dict(zip(names, params, strict=False)), (a,))
        return ((out - b) ** 2).mean()

    uu = tuple(u[i] for gr in groups for i in gr)
    loss = full_loss(x, y)
    return compute_Hg(ps, loss, uu)


def test_subsetwise_orthogonal_reparameterization_leaves_the_step_unchanged(f64):
    """NOT in Appendix E. Every summary is pointwise invariant under
    theta_s -> Q_s theta_s with Q_s orthogonal, hence eta is *unchanged* (not
    merely rescaled). The exact group is (scalar per subset) x (orthogonal per
    subset), not the `affine` of the appendix title."""
    base = _uniform_mlp(width=5, depth=3)
    names = [n for n, _ in base.named_parameters()]
    P0 = [p.detach().clone().requires_grad_(True) for _, p in base.named_parameters()]
    U0 = [torch.randn_like(p) for p in P0]
    x, y = torch.randn(20, 5), torch.randn(20, 5)
    groups = [[i] for i in range(len(P0))]

    H0, g0, o0 = _summaries(base, names, P0, groups, U0, x, y)

    Q = [torch.linalg.qr(torch.randn(p.shape[0], p.shape[0]))[0] for p in P0]
    P1 = [(q.T @ p).detach().clone().requires_grad_(True) for q, p in zip(Q, P0, strict=False)]
    U1 = [q.T @ u for q, u in zip(Q, U0, strict=False)]
    ps1 = ParamStructure([{"params": [p]} for p in P1])

    def full_loss1(a, b):
        real = [q @ p for q, p in zip(Q, P1, strict=False)]
        out = torch.func.functional_call(base, dict(zip(names, real, strict=False)), (a,))
        return ((out - b) ** 2).mean()

    loss = full_loss1(x, y)
    H1, g1, o1 = compute_Hg(ps1, loss, tuple(U1))

    assert torch.allclose(g1, g0, rtol=1e-9, atol=1e-12)
    assert torch.allclose(H1, H0, rtol=1e-9, atol=1e-12)
    assert torch.allclose(o1, o0, rtol=1e-8, atol=1e-11)

    e0, l0 = nesterov_lrs(H0, g0, o0.abs().pow(1 / 3), damping_int=1.0)
    e1, l1 = nesterov_lrs(H1, g1, o1.abs().pow(1 / 3), damping_int=1.0)
    assert l0["found"] and l1["found"]
    assert torch.allclose(e1, e0, rtol=1e-8, atol=1e-11)


def test_invariance_requires_one_scale_per_group_not_per_tensor(f64):
    """The unwritten precondition of Appendix E: the Jacobian must be a_s * I on
    the WHOLE group. Scaling a weight and its bias differently breaks it."""
    base = _uniform_mlp(width=5, depth=3)
    names = [n for n, _ in base.named_parameters()]
    P0 = [p.detach().clone() for _, p in base.named_parameters()]
    U0 = [torch.randn_like(p) for p in P0]
    x, y = torch.randn(20, 5), torch.randn(20, 5)
    groups = [[0, 1], [2, 3], [4, 5]]  # one layer (weight + bias) per group

    def run(scales):
        Pt = [
            (p / a).detach().clone().requires_grad_(True) for p, a in zip(P0, scales, strict=False)
        ]
        Ut = [a * u for u, a in zip(U0, scales, strict=False)]
        ps = ParamStructure([{"params": [Pt[i] for i in gr]} for gr in groups])

        def full_loss(a_, b_):
            real = [s * p for s, p in zip(scales, Pt, strict=False)]
            out = torch.func.functional_call(base, dict(zip(names, real, strict=False)), (a_,))
            return ((out - b_) ** 2).mean()

        uu = tuple(Ut[i] for gr in groups for i in gr)
        loss = full_loss(x, y)
        H, g, o = compute_Hg(ps, loss, uu)
        e, log = nesterov_lrs(H, g, o.abs().pow(1 / 3), damping_int=1.0)
        assert log["found"]
        return e

    ref = run([1.0] * 6)

    common = [2.0, 2.0, 0.5, 0.5, 3.0, 3.0]
    A = torch.tensor([4.0, 0.25, 9.0], dtype=torch.float64)
    assert torch.allclose(run(common) * A, ref, rtol=1e-7, atol=1e-10)

    split = [2.0, 0.5, 0.5, 3.0, 3.0, 2.0]
    got = run(split)
    assert not torch.allclose(got * A, ref, rtol=1e-2), (
        "a per-tensor scale inside a group must NOT be absorbed by eta -> eta/a_s^2"
    )
