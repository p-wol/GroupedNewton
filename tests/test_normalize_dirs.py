"""Tests for `optimizer.hg.normalize_dirs`.

Three levels, from cheapest to most meaningful:

  1. the postcondition -- ||u_s|| == 1 on every subset, for every partition;
  2. the contract of `ParamStructure.expand_src_as_params`, whose misuse is what
     broke (1) for every partition but `canonical`;
  3. the invariant the feature exists to restore -- averaging (Hbar, gbar, D)
     across steps with different ||u^(t)|| is not scale-free, and normalizing
     makes it so.

(3) is the one that would justify the option in a paper; (1) and (2) are the ones
that fail loudly when the plumbing is wrong.
"""

import pytest
import torch

from grnewt import (
    NewtonSummaryStaticAvg,
    NewtonSummaryUniformAvg,
    ParamStructure,
    optimizers,
)
from grnewt import partition as build_partition
from grnewt.config import HgCfg, NesterovCfg, StaticAvgCfg, UniformAvgCfg

PARTITIONS = {
    "canonical": build_partition.canonical,  # S == P: the only case that worked
    "trivial": build_partition.trivial,  # S == 1
    "wb": build_partition.wb,  # S == 2, reorders
    "blocks-2": lambda m: build_partition.blocks(m, 2),
}


def _model():
    return torch.nn.Sequential(
        torch.nn.Linear(6, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 3),
    )


def _problem(seed=0, n=96, bs=12):
    torch.manual_seed(seed)
    model = _model()
    x = torch.randn(n, 6)
    y = torch.randint(0, 3, (n,))
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    return model, loader, full_loss


# ---------------------------------------------------------------------------
# 2. the contract that was misused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PARTITIONS))
def test_expand_src_as_params_has_one_entry_per_tensor(f64, name):
    model = _model()
    pgroups, _ = PARTITIONS[name](model)
    ps = ParamStructure(pgroups)

    per_group = torch.arange(1.0, ps.nb_groups + 1.0, dtype=torch.float64)
    expanded = ps.expand_src_as_params(per_group)

    assert len(expanded) == len(ps.tup_params)
    # entry i must carry the value of the subset that parameter i belongs to
    expected = [float(per_group[s]) for s, group in enumerate(ps.pgroups) for _ in group["params"]]
    assert [float(v) for v in expanded] == expected


# ---------------------------------------------------------------------------
# 1. the postcondition
# ---------------------------------------------------------------------------


def _norms(ps, direction):
    return ps.squared_norm(direction).sqrt()


@pytest.mark.parametrize("name", sorted(PARTITIONS))
def test_normalize_dirs_gives_every_subset_unit_norm(f64, name):
    """Fails with `ValueError: zip() argument 2 is shorter than argument 1` for
    every partition but `canonical` if `expand_src_as_params`'s return value is
    dropped, and passes on `canonical` only because S == P there."""
    model, loader, full_loss = _problem()
    pgroups, _ = PARTITIONS[name](model)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    cfg = HgCfg(
        period_hg=2,
        normalize_dirs=True,
        nesterov=NesterovCfg(use=True, damping_int=1.0),
        static_avg=StaticAvgCfg(nsamples=1),
    )
    opt = NewtonSummaryStaticAvg(
        pgroups,
        full_loss,
        loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )

    seen = []
    real = opt.compute_avg_Hg

    def spy(direction):
        seen.append(_norms(opt.param_struct, direction).clone())
        return real(direction)

    opt.compute_avg_Hg = spy

    it = iter(loader)
    for _ in range(4):
        x, y = next(it)
        updater.zero_grad()
        full_loss(x, y).backward()
        opt.step()

    assert seen, "compute_avg_Hg was never reached"
    for n in seen:
        assert n.numel() == len(pgroups)
        assert torch.allclose(n, torch.ones_like(n), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(PARTITIONS))
def test_directions_are_untouched_when_the_option_is_off(f64, name):
    model, loader, full_loss = _problem()
    pgroups, _ = PARTITIONS[name](model)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    cfg = HgCfg(
        period_hg=2,
        normalize_dirs=False,
        nesterov=NesterovCfg(use=True, damping_int=1.0),
        static_avg=StaticAvgCfg(nsamples=1),
    )
    opt = NewtonSummaryStaticAvg(
        pgroups,
        full_loss,
        loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )
    seen = []
    real = opt.compute_avg_Hg
    opt.compute_avg_Hg = lambda d: (seen.append(_norms(opt.param_struct, d).clone()), real(d))[1]

    x, y = next(iter(loader))
    updater.zero_grad()
    full_loss(x, y).backward()
    opt.step()

    assert seen and not torch.allclose(seen[0], torch.ones_like(seen[0]), atol=1e-6)


def test_a_zero_direction_does_not_produce_nan(f64):
    """A frozen or dead subset has ||u_s|| == 0. Dividing gives nan, which then
    silently poisons Hbar and every downstream log. The subset must be left as
    the zero vector, which nesterov_lrs already reports as infeasible_kernel."""
    model, loader, full_loss = _problem()
    pgroups, _ = build_partition.canonical(model)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    cfg = HgCfg(period_hg=1, normalize_dirs=True, nesterov=NesterovCfg(use=True))
    opt = NewtonSummaryStaticAvg(
        pgroups,
        full_loss,
        loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )
    ps = opt.param_struct
    direction = tuple(torch.randn_like(p) for p in ps.tup_params)
    direction[0].zero_()  # subset 0 is dead

    opt.normalize_dirs_(direction)

    assert torch.isfinite(torch.cat([d.reshape(-1) for d in direction])).all()
    n = _norms(ps, direction)
    assert float(n[0]) == 0.0
    assert torch.allclose(n[1:], torch.ones_like(n[1:]), atol=1e-12)


# ---------------------------------------------------------------------------
# 3. the invariant the option exists to restore
# ---------------------------------------------------------------------------


def _scaled_run(normalize, scale_fn, n_steps=18, seed=3):
    """Run NSUA while rescaling each subset of `direction` by a per-step,
    per-subset factor. `scale_fn(step, s) -> float`."""
    model, loader, full_loss = _problem(seed=seed)
    pgroups, _ = build_partition.canonical(model)
    ps_ref = ParamStructure(pgroups)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    cfg = HgCfg(
        damping=0.1,
        period_hg=2,
        mom_lrs=0.0,
        remove_negative=True,
        normalize_dirs=normalize,
        nesterov=NesterovCfg(use=True, damping_int=1.0),
        uniform_avg=UniformAvgCfg(period=3, warmup=0),
    )
    opt = NewtonSummaryUniformAvg(
        pgroups,
        full_loss,
        loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )

    # inject the rescaling between the updater and NSBase.step
    raw = updater.compute_step
    state = {"t": 0}

    def scaled(*a, **k):
        out = raw(*a, **k)
        by_group = ps_ref.expand_src_as_params(
            torch.tensor(
                [scale_fn(state["t"], s) for s in range(ps_ref.nb_groups)], dtype=ps_ref.dtype
            )
        )
        perm = ps_ref.build_reindex([p for gr in updater.param_groups for p in gr["params"]])
        ordered = ps_ref.reindex(out, perm)
        for d, c in zip(ordered, by_group, strict=True):
            d.mul_(c)
        state["t"] += 1
        return out

    updater.compute_step = scaled

    it = iter(loader)
    for _ in range(n_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        updater.zero_grad()
        full_loss(x, y).backward()
        opt.step()
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _identity(_t, _s):
    return 1.0


def _time_varying(t, s):
    # per-subset AND per-step, so successive (Hbar, gbar, D) carry different C_t
    return float(1.0 + 0.5 * ((t + s) % 3))


def test_normalization_makes_the_averaged_summaries_scale_free(f64):
    """u_s -> c_s(t) u_s must not change the trajectory when normalization is on.

    A single (Hbar, gbar, D) triple is already scale-free: eta_s -> eta_s / c_s
    and the step eta_s u_s is exactly invariant. The invariance breaks once
    triples from steps with different C_t are averaged, which is what
    NewtonSummaryUniformAvg does. Normalizing restores it.
    """
    ref = _scaled_run(normalize=True, scale_fn=_identity)
    got = _scaled_run(normalize=True, scale_fn=_time_varying)
    # Not bitwise: u/||u|| and (cu)/||cu|| are mathematically equal but round
    # differently, and 18 steps of a cubic solve amplify that. Measured 8.0e-15
    # against 2.2e+00 of trajectory, i.e. ~4e-15 relative -- rounding, not drift.
    # The un-normalized control below differs by ~1e-1, four orders of magnitude
    # above any tolerance one could argue about.
    assert torch.allclose(ref, got, rtol=1e-10, atol=1e-12), (
        f"max|diff| = {float((ref - got).abs().max()):.3e}; the normalized run must "
        "not depend on the scale of the proposed direction"
    )


def test_without_normalization_the_averaged_summaries_are_not_scale_free(f64):
    """The control. If this ever passes, the test above proves nothing."""
    ref = _scaled_run(normalize=False, scale_fn=_identity)
    got = _scaled_run(normalize=False, scale_fn=_time_varying)
    assert not torch.allclose(ref, got, rtol=1e-6, atol=1e-8), (
        "a per-step, per-subset rescaling of the direction left the un-normalized "
        "run unchanged; the scale-freeness defect this option fixes is not being "
        "exercised"
    )


def test_a_constant_rescaling_is_harmless_even_without_normalization(f64):
    """Sharpens the claim: it is the TIME variation of ||u^(t)||, not its value,
    that breaks the average. A constant C_t is absorbed exactly."""
    ref = _scaled_run(normalize=False, scale_fn=_identity)
    got = _scaled_run(normalize=False, scale_fn=lambda t, s: 1.0 + 0.5 * s)
    assert torch.allclose(ref, got, rtol=1e-9, atol=1e-11), (
        f"max|diff| = {float((ref - got).abs().max()):.3e}"
    )
