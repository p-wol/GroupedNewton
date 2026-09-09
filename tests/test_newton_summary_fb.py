"""Tests for the rewritten `NewtonSummaryFB`.

NSFB is the only optimizer whose summaries are exact rather than estimated: it
sweeps the whole training set for the direction and again for (Hbar, gbar, order3).
So it can be checked against a ground truth that no stochastic variant has --
the single-batch computation on the full dataset -- and that is what most of this
file does.

Four properties, in order of how much they would cost if they broke:
  1. the accumulation is the full-batch quantity, for every partition and every
     config branch (this is the whole point of the optimizer);
  2. it is independent of the batch size of the loader, and refuses loaders that
     do not cover the dataset exactly once;
  3. it does not disturb the caller's iteration of the training loader;
  4. on a quadratic it is exact.
"""

import pytest
import torch

from grnewt import (
    NewtonSummaryFB,
    ParamStructure,
    compute_Hg,
)
from grnewt import (
    partition as build_partition,
)
from grnewt.config import HgCfg, NesterovCfg
from grnewt.optimizers import FBGDUpdate


def _pre(x, y):
    return x, y


def _model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(6, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 3),
    )


def _data(n=96, seed=0):
    torch.manual_seed(seed + 1000)
    x = torch.randn(n, 6)
    y = torch.randint(0, 3, (n,))
    return x, y, torch.utils.data.TensorDataset(x, y)


PARTITIONS = {
    "canonical": build_partition.canonical,
    "trivial": build_partition.trivial,
    "wb": build_partition.wb,  # reorders w.r.t. model.parameters()
    "blocks-2": lambda m: build_partition.blocks(m, 2),
}


def _make(model, dataset, n=None, batch_size=12, partition="canonical", **cfg_kw):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    pgroups, _ = PARTITIONS[partition](model)
    cfg_kw.setdefault("nesterov", NesterovCfg(use=True, damping_int=1.0))
    opt = NewtonSummaryFB(
        pgroups,
        full_loss,
        model,
        loss_fn,
        loader,
        loader_pre_hook=_pre,
        cfg=HgCfg(**cfg_kw),
    )
    return opt, loader, full_loss, ParamStructure(pgroups)


# ---------------------------------------------------------------------------
# 1. the accumulation IS the full-batch quantity
# ---------------------------------------------------------------------------


def test_fullbatch_gradient_matches_a_single_batch_backward(f64):
    model = _model()
    x, y, ds = _data()
    loader = torch.utils.data.DataLoader(ds, batch_size=12)
    loss_fn = torch.nn.CrossEntropyLoss()

    got = FBGDUpdate(model, loss_fn, loader, loader_pre_hook=_pre).compute_step()

    model.zero_grad()
    loss_fn(model(x), y).backward()
    want = tuple(p.grad.clone() for p in model.parameters())

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("partition", sorted(PARTITIONS))
def test_summaries_match_the_whole_dataset_in_one_batch(f64, partition):
    """The ground truth NSFB exists to reproduce."""
    model = _model()
    x, y, ds = _data()
    opt, _, full_loss, ps = _make(model, ds, len(ds), partition=partition)

    perm = ps.build_reindex(list(model.parameters()))
    u = ps.reindex(tuple(torch.randn_like(p) for p in model.parameters()), perm)

    H, g, order3, instr = opt.compute_avg_Hg(u)
    H_ref, g_ref, o3_ref = compute_Hg(ps, full_loss, x, y, u)

    assert instr.recompute_lrs and instr.do_update
    assert torch.allclose(H, H_ref, rtol=1e-11, atol=1e-13)
    assert torch.allclose(g, g_ref, rtol=1e-11, atol=1e-13)
    assert torch.allclose(order3, o3_ref, rtol=1e-10, atol=1e-12)
    assert H.shape == (ps.nb_groups, ps.nb_groups)


@pytest.mark.parametrize(
    "cfg_kw",
    [
        pytest.param(dict(noregul=True, nesterov=NesterovCfg(use=False)), id="noregul"),
        pytest.param(dict(diagonal=True), id="diagonal"),
        pytest.param(dict(hg_batched=True, hg_batched_chunk=1), id="batched-1"),
        pytest.param(dict(hg_batched=True, hg_batched_chunk=-1), id="batched-all"),
    ],
)
def test_every_config_branch_reaches_the_same_full_batch_value(f64, cfg_kw):
    model = _model()
    x, y, ds = _data()
    opt, _, full_loss, ps = _make(model, ds, len(ds), **cfg_kw)
    u = tuple(torch.randn_like(p) for p in ps.tup_params)

    H, g, order3, _ = opt.compute_avg_Hg(u)
    ref_kw = {k: cfg_kw[k] for k in ("noregul", "diagonal") if k in cfg_kw}
    H_ref, g_ref, o3_ref = compute_Hg(ps, full_loss, x, y, u, **ref_kw)

    assert torch.allclose(H, H_ref, rtol=1e-11, atol=1e-13)
    assert torch.allclose(g, g_ref, rtol=1e-11, atol=1e-13)
    assert torch.allclose(order3, o3_ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. batch size must not be a hyperparameter of the result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 7, 12, 96])
def test_summaries_do_not_depend_on_the_batch_size(f64, batch_size):
    """7 does not divide 96, so this also covers the ragged last batch: the
    per-batch weight is n_b / train_size, not 1 / n_batches."""
    model = _model()
    x, y, ds = _data(n=96)
    opt, _, full_loss, ps = _make(model, ds, len(ds), batch_size=batch_size)
    u = tuple(torch.randn_like(p) for p in ps.tup_params)

    H, g, order3, _ = opt.compute_avg_Hg(u)
    H_ref, g_ref, o3_ref = compute_Hg(ps, full_loss, x, y, u)

    assert torch.allclose(H, H_ref, rtol=1e-10, atol=1e-13)
    assert torch.allclose(g, g_ref, rtol=1e-10, atol=1e-13)
    assert torch.allclose(order3, o3_ref, rtol=1e-9, atol=1e-12)


def test_a_loader_that_drops_the_tail_is_refused(f64):
    """`drop_last=True` biases Hbar, gbar and order3 by three DIFFERENT factors
    (measured 0.987 / 0.738 / 1.096 on a loader covering 90/100 samples), so it
    is not even absorbable as a rescaling of damping_int."""
    model = _model()
    _, _, ds = _data(n=100)
    loader = torch.utils.data.DataLoader(ds, batch_size=30, drop_last=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    pgroups, _ = build_partition.canonical(model)
    with pytest.raises(ValueError, match="drop_last"):
        NewtonSummaryFB(
            pgroups,
            lambda a, b: loss_fn(model(a), b),
            model,
            loss_fn,
            loader,
            loader_pre_hook=_pre,
            cfg=HgCfg(),
        )


def test_train_size_is_derived_from_the_loader_not_passed_in(f64):
    """It used to be a separate argument that could silently disagree with the loader.
    Deriving it makes the inconsistency unrepresentable."""
    model = _model()
    _, _, ds = _data(n=96)
    opt, *_ = _make(model, ds, batch_size=12)
    assert opt.train_size == len(ds) == 96
    assert opt.updater.train_size == 96


def test_a_parameter_without_gradient_gives_zero_not_an_exception(f64):
    """An unused head keeps grad=None; `p.grad.clone()` used to raise
    AttributeError from inside the optimizer."""
    model = _model()
    model.unused = torch.nn.Parameter(torch.zeros(4))
    _, _, ds = _data()
    loader = torch.utils.data.DataLoader(ds, batch_size=12)
    grad = FBGDUpdate(
        model, torch.nn.CrossEntropyLoss(), loader, loader_pre_hook=_pre
    ).compute_step()
    params = list(model.parameters())
    assert len(grad) == len(params)
    # `unused` is registered directly on the Sequential, so parameters() yields it
    # first, not last: locate it by identity rather than by position.
    idx = next(i for i, p in enumerate(params) if p is model.unused)
    assert torch.equal(grad[idx], torch.zeros(4, dtype=torch.float64))
    assert any(g.abs().sum() > 0 for i, g in enumerate(grad) if i != idx)


# ---------------------------------------------------------------------------
# 3. NSFB must not disturb the caller's iteration
# ---------------------------------------------------------------------------


class _CountingLoader(torch.utils.data.DataLoader):
    """Counts how many times someone starts iterating this loader object."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_iter = 0

    def __iter__(self):
        self.n_iter += 1
        return super().__iter__()


def test_step_never_iterates_the_loader_the_caller_holds(f64):
    """The structural invariant behind the bug, tested without worker processes.

    `training_hydra.train_epoch` calls `step()` from inside its own
    `for ... in self.train_loader`. NSFB sweeps the training set twice per step,
    so if it sweeps the caller's loader object it re-enters an active iterator.
    With persistent_workers=True that silently truncated the caller's epoch to a
    single minibatch -- 1 batch instead of 4, no exception, no warning -- and
    datasets.py sets persistent_workers = num_workers > 0 in production.

    Asserting on the object rather than on the symptom keeps this deterministic:
    reproducing the truncation itself needs real worker processes and is not a
    stable thing to put in CI.
    """
    n, batch_size = 100, 25
    model = _model()
    _, _, ds = _data(n=n)
    caller_loader = _CountingLoader(ds, batch_size=batch_size)
    fb_loader = _CountingLoader(ds, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    pgroups, _ = build_partition.canonical(model)
    opt = NewtonSummaryFB(
        pgroups,
        full_loss,
        model,
        loss_fn,
        fb_loader,
        loader_pre_hook=_pre,
        cfg=HgCfg(nesterov=NesterovCfg(use=True, damping_int=1.0)),
    )
    assert caller_loader.n_iter == 0 and fb_loader.n_iter == 0

    seen = 0
    for x, y in caller_loader:
        seen += 1
        model.zero_grad()
        full_loss(x, y).backward()
        opt.step()

    assert seen == n // batch_size
    assert caller_loader.n_iter == 1, (
        f"NewtonSummaryFB started {caller_loader.n_iter - 1} extra iteration(s) of the "
        "loader the caller is iterating"
    )
    # two sweeps per step: one for the full-batch gradient, one for the summaries
    assert fb_loader.n_iter == 2 * seen


def test_the_fb_loader_must_not_be_the_training_loader(f64):
    """Nothing stops a caller from passing the same object twice, so at least make
    the cost visible: `step()` sweeps fb_loader twice, and training_hydra calls
    `step()` once per minibatch, i.e. 2*(N/B) epochs of data per nominal epoch."""
    n, batch_size = 100, 25
    model = _model()
    _, _, ds = _data(n=n)
    fb_loader = _CountingLoader(ds, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    pgroups, _ = build_partition.canonical(model)
    opt = NewtonSummaryFB(
        pgroups,
        full_loss,
        model,
        loss_fn,
        fb_loader,
        loader_pre_hook=_pre,
        cfg=HgCfg(nesterov=NesterovCfg(use=True, damping_int=1.0)),
    )
    x, y = next(iter(torch.utils.data.DataLoader(ds, batch_size=batch_size)))
    fb_loader.n_iter = 0
    model.zero_grad()
    full_loss(x, y).backward()
    opt.step()
    assert fb_loader.n_iter == 2


@pytest.mark.parametrize(
    "loader_kw",
    [pytest.param(dict(), id="workers-0"), pytest.param(dict(shuffle=True), id="shuffle")],
)
def test_the_callers_epoch_is_not_truncated(f64, loader_kw):
    """End-to-end version of the invariant above, worker-free so that it is stable
    in CI. The worker-process variant is what actually failed in production; it is
    covered structurally by test_step_never_iterates_the_loader_the_caller_holds."""
    n, batch_size = 100, 25
    model = _model()
    _, _, ds = _data(n=n)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, **loader_kw)
    fb_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    pgroups, _ = build_partition.canonical(model)
    opt = NewtonSummaryFB(
        pgroups,
        full_loss,
        model,
        loss_fn,
        fb_loader,
        loader_pre_hook=_pre,
        cfg=HgCfg(nesterov=NesterovCfg(use=True, damping_int=1.0)),
    )
    seen = 0
    for x, y in loader:
        seen += 1
        model.zero_grad()
        full_loss(x, y).backward()
        opt.step()
    assert seen == n // batch_size


# ---------------------------------------------------------------------------
# 4. exactness and determinism
# ---------------------------------------------------------------------------


def test_one_step_is_the_exact_line_minimiser_on_a_quadratic(f64):
    """S = 1 (trivial partition), noregul, damping 1: eta = gbar / Hbar is the
    exact minimiser of t -> L(theta - t*u) when L is quadratic in theta."""
    torch.manual_seed(4)
    n, d = 120, 7
    a = torch.randn(n, d)
    b = torch.randn(n)
    lin = torch.nn.Linear(d, 1, bias=False)
    ds = torch.utils.data.TensorDataset(a, b)
    loader = torch.utils.data.DataLoader(ds, batch_size=20)

    def mse(out, t):
        return ((out.squeeze(-1) - t) ** 2).mean()

    def full_loss(x, t):
        return mse(lin(x), t)

    pgroups, _ = build_partition.trivial(lin)
    opt = NewtonSummaryFB(
        pgroups,
        full_loss,
        lin,
        mse,
        loader,
        loader_pre_hook=_pre,
        cfg=HgCfg(
            damping=1.0, noregul=True, remove_negative=False, nesterov=NesterovCfg(use=False)
        ),
    )

    w0 = lin.weight.detach().clone()
    lin.zero_grad()
    full_loss(a, b).backward()
    g_full = lin.weight.grad.detach().clone()

    opt.step()
    t_taken = float(((w0 - lin.weight.detach()) / g_full).flatten()[0])

    # closed form: L(w0 - t g) is quadratic in t, minimised at g'g / (g' A g)
    with torch.no_grad():
        hess_g = 2.0 * (a.T @ (a @ g_full.flatten())) / n
        t_star = float((g_full.flatten() @ g_full.flatten()) / (g_full.flatten() @ hess_g))
    assert t_taken == pytest.approx(t_star, rel=1e-10)

    with torch.no_grad():
        lin.weight.data.copy_(w0 - t_star * g_full)
        l_star = float(full_loss(a, b))
        for eps in (-1e-3, 1e-3):
            lin.weight.data.copy_(w0 - (t_star + eps) * g_full)
            assert float(full_loss(a, b)) >= l_star


@pytest.mark.parametrize("shuffle", [False, True])
def test_two_identical_runs_agree_bitwise(f64, shuffle):
    """NSFB draws no minibatch of its own, so shuffling the loader may only
    change the summation order of an exact full-batch sum."""

    def run():
        model = _model(seed=7)
        x, y, ds = _data(n=96, seed=7)
        loader = torch.utils.data.DataLoader(ds, batch_size=24, shuffle=shuffle)
        loss_fn = torch.nn.CrossEntropyLoss()

        def full_loss(a, b):
            return loss_fn(model(a), b)

        pgroups, _ = build_partition.canonical(model)
        opt = NewtonSummaryFB(
            pgroups,
            full_loss,
            model,
            loss_fn,
            loader,
            loader_pre_hook=_pre,
            cfg=HgCfg(damping=0.1, nesterov=NesterovCfg(use=True, damping_int=1.0)),
        )
        for _ in range(3):
            model.zero_grad()
            full_loss(x, y).backward()
            opt.step()
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    assert torch.equal(run(), run())


def test_the_direction_reaches_compute_Hg_in_tup_params_order(f64, monkeypatch):
    """The producer here is FBGDUpdate, which emits in model.parameters() order;
    `compute_Hg` consumes in partition order. `ParamStructure.dot` broadcasts
    rather than raising, so a permuted direction is silently wrong."""
    import grnewt.newton_summary_fb as mod

    model = _model()
    _, _, ds = _data()
    seen = {}
    orig = mod.compute_Hg

    def spy(param_struct, full_loss, x, y, direction, **kw):
        seen["direction"] = direction
        return orig(param_struct, full_loss, x, y, direction, **kw)

    monkeypatch.setattr(mod, "compute_Hg", spy)
    opt, loader, full_loss, ps = _make(model, ds, len(ds), partition="wb")
    x, y = next(iter(loader))
    model.zero_grad()
    full_loss(x, y).backward()
    opt.step()

    direction = seen["direction"]
    assert len(direction) == len(ps.tup_params)
    for dvec, p in zip(direction, ps.tup_params, strict=True):
        assert dvec.shape == p.shape


def test_it_reduces_the_loss(f64):
    model = _model(seed=3)
    x, y, ds = _data(n=120, seed=3)
    opt, _, full_loss, _ = _make(
        model, ds, len(ds), batch_size=30, damping=0.3, remove_negative=True
    )
    before = float(full_loss(x, y))
    for _ in range(8):
        model.zero_grad()
        full_loss(x, y).backward()
        opt.step()
    after = float(full_loss(x, y))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert after < before, f"{before:.5f} -> {after:.5f}"


# ---------------------------------------------------------------------------
# 5. the call site: one step per epoch, not one per minibatch
# ---------------------------------------------------------------------------


def test_one_step_costs_exactly_two_sweeps(f64):
    """The cost model, pinned. `Trainer.step_train_fb` calls step() ONCE per epoch, and
    step() sweeps fb_loader twice: once for the full-batch gradient, once for the
    summaries. Both sweeps must use the same loader, so that under BatchNorm they see
    the same batch partition."""
    n, batch_size = 100, 25
    model = _model()
    _, _, ds = _data(n=n)
    fb_loader = _CountingLoader(ds, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    pgroups, _ = build_partition.canonical(model)
    opt = NewtonSummaryFB(
        pgroups,
        full_loss,
        model,
        loss_fn,
        fb_loader,
        loader_pre_hook=_pre,
        cfg=HgCfg(nesterov=NesterovCfg(use=True, damping_int=1.0)),
    )
    opt.step()
    assert fb_loader.n_iter == 2
    assert len(opt.logs["H"]) == 1
    assert opt.updater.loader is opt.fb_loader
