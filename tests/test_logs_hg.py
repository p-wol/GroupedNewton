"""The per-epoch full-batch diagnostic: `NSBase.probe()` and the `logs_hg` node.

The diagnostic runs a second `NewtonSummaryFB` in observation mode so that the
logged (Hbar, gbar, order3, lrs) come from exactly the code path the optimizer
uses. Two things have to hold for that to be worth anything:

  * observing must not perturb -- no parameter, no lr, no momentum state, no
    counter, no log list may move;
  * the observer must be configurable independently of the run, and its config
    must go through the same validation.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from grnewt import (
    NewtonSummaryFB,
    NewtonSummaryStaticAvg,
    optimizers,
)
from grnewt import (
    partition as build_partition,
)
from grnewt.config import HgCfg, LogsCfg, NesterovCfg, StaticAvgCfg, from_dictconfig_logs_hg


def _pre(x, y):
    return x, y


def _problem(n=96, batch_size=24, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    x = torch.randn(n, 6)
    y = torch.randint(0, 3, (n,))
    ds = TensorDataset(x, y)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    return model, ds, loss_fn, full_loss, DataLoader(ds, batch_size=batch_size)


def _logger(model, ds, loss_fn, full_loss, *, partition="trivial", batch_size=24, **cfg_kw):
    pgroups, _ = getattr(build_partition, partition)(model)
    cfg_kw.setdefault("nesterov", NesterovCfg(use=True, damping_int=1.0))
    return NewtonSummaryFB(
        pgroups,
        full_loss,
        model,
        loss_fn,
        DataLoader(ds, batch_size=batch_size),
        loader_pre_hook=_pre,
        cfg=LogsCfg(use=True, **cfg_kw),
    )


# ---------------------------------------------------------------------------
# observing must not perturb
# ---------------------------------------------------------------------------


def test_probe_touches_nothing(f64):
    """This is why `step(dry_run=True)` was not enough: it suppressed only the
    parameter update, while group["lr"], curr_lrs, step_counter and self.logs were
    still written. A logger sharing that path perturbs the run it observes."""
    model, ds, loss_fn, full_loss, _ = _problem()
    logger = _logger(model, ds, loss_fn, full_loss)

    before_params = [p.detach().clone() for p in model.parameters()]
    before_lr = [g["lr"] for g in logger.param_groups]
    before_damping = [g["damping"] for g in logger.param_groups]
    before_counter = logger.step_counter
    before_curr = logger.curr_lrs
    before_logs = {k: len(v) for k, v in logger.logs.items()}

    logger.probe()

    for a, b in zip(model.parameters(), before_params, strict=True):
        assert torch.equal(a.detach(), b)
    assert [g["lr"] for g in logger.param_groups] == before_lr
    assert [g["damping"] for g in logger.param_groups] == before_damping
    assert logger.step_counter == before_counter
    assert logger.curr_lrs is before_curr or logger.curr_lrs == before_curr
    assert {k: len(v) for k, v in logger.logs.items()} == before_logs


def test_probe_is_idempotent(f64):
    """No hidden state: probing twice at the same point gives the same numbers."""
    model, ds, loss_fn, full_loss, _ = _problem()
    logger = _logger(model, ds, loss_fn, full_loss, partition="canonical")
    a = logger.probe()
    b = logger.probe()
    for key in ("H", "g", "order3", "lrs"):
        assert torch.equal(a[key], b[key]), key


def test_probe_returns_something_torch_save_can_write(f64):
    """`train()` writes the dict straight to Hg_logs_ext.<epoch>.pkl."""
    model, ds, loss_fn, full_loss, _ = _problem()
    logger = _logger(model, ds, loss_fn, full_loss, partition="canonical")
    logs = logger.probe()

    assert {"H", "g", "order3", "lrs"} <= set(logs)
    assert logs["H"].shape == (logger.param_struct.nb_groups,) * 2
    assert logs["g"].shape == (logger.param_struct.nb_groups,)
    assert logs["lrs"].shape == (logger.param_struct.nb_groups,)
    assert logs["nesterov.found"] is True
    assert all(not torch.is_tensor(v) or v.grad_fn is None for v in logs.values())


def test_the_logger_may_use_a_different_partition_from_the_run(f64):
    """The point of giving logs_hg its own `partition`: report the trivial-partition
    step while the run itself uses `canonical`."""
    model, ds, loss_fn, full_loss, loader = _problem()

    pg_opt, _ = build_partition.canonical(model)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    opt = NewtonSummaryStaticAvg(
        pg_opt,
        full_loss,
        loader,
        updater,
        loader_pre_hook=_pre,
        cfg=HgCfg(
            damping=0.1,
            period_hg=2,
            nesterov=NesterovCfg(use=True, damping_int=1.0),
            static_avg=StaticAvgCfg(nsamples=2),
        ),
    )
    logger = _logger(model, ds, loss_fn, full_loss, partition="trivial")

    assert logger.param_struct.nb_groups == 1
    assert opt.param_struct.nb_groups == len(list(model.parameters()))

    it = iter(loader)
    for _ in range(4):
        logs = logger.probe()
        assert logs["H"].shape == (1, 1)
        x, y = next(it)
        updater.zero_grad()
        full_loss(x, y).backward()
        opt.step()

    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_probing_does_not_change_the_trajectory(f64):
    """The decisive end-to-end version: a run with the diagnostic on must be
    bitwise identical to the same run with it off."""

    def run(with_logger):
        model, ds, loss_fn, full_loss, loader = _problem(seed=3)
        pg_opt, _ = build_partition.canonical(model)
        updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
        opt = NewtonSummaryStaticAvg(
            pg_opt,
            full_loss,
            loader,
            updater,
            loader_pre_hook=_pre,
            cfg=HgCfg(
                damping=0.1,
                period_hg=2,
                nesterov=NesterovCfg(use=True, damping_int=1.0),
                static_avg=StaticAvgCfg(nsamples=2),
            ),
        )
        logger = _logger(model, ds, loss_fn, full_loss) if with_logger else None
        it = iter(loader)
        for _ in range(6):
            if logger is not None:
                logger.probe()
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            updater.zero_grad()
            full_loss(x, y).backward()
            opt.step()
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    assert torch.equal(run(False), run(True))


def test_probe_agrees_with_step_on_the_same_quantities(f64):
    """probe() and step() must compute (Hbar, gbar, order3) with the same code, not
    with two implementations that can drift -- which is exactly what happened to the
    hand-rolled compute_logs_hg, left on an obsolete compute_Hg signature."""
    model, ds, loss_fn, full_loss, _ = _problem()
    logger = _logger(model, ds, loss_fn, full_loss, partition="canonical")

    probed = logger.probe()
    logger.step()

    assert torch.allclose(logger.logs["H"][0], probed["H"], rtol=1e-12, atol=1e-14)
    assert torch.allclose(logger.logs["g"][0], probed["g"], rtol=1e-12, atol=1e-14)
    assert torch.allclose(logger.logs["order3"][0], probed["order3"], rtol=1e-12, atol=1e-14)


def test_probe_reports_a_failed_solve_as_lrs_None(f64):
    """`torch.save` must not be handed a half-written dict when the reduced problem
    has no solution."""
    model, ds, loss_fn, full_loss, _ = _problem()
    logger = _logger(
        model,
        ds,
        loss_fn,
        full_loss,
        partition="canonical",
        nesterov=NesterovCfg(use=True, damping_int=0.0),
    )
    logs = logger.probe()
    assert (logs["lrs"] is None) == (not logs["nesterov.found"])
    assert {"H", "g", "order3"} <= set(logs)


# ---------------------------------------------------------------------------
# the logs_hg config node
# ---------------------------------------------------------------------------


def test_logs_cfg_is_an_hg_cfg_plus_use(f64):
    """One schema, not two. A parallel LogsCfg would have to be kept in sync with
    every field added to HgCfg, and check_consumed would not apply to it."""
    import dataclasses

    from omegaconf import OmegaConf

    hg_names = {f.name for f in dataclasses.fields(HgCfg)}
    logs_names = {f.name for f in dataclasses.fields(LogsCfg)}
    assert hg_names < logs_names
    assert logs_names - hg_names == {"use"}

    node = OmegaConf.create(
        {
            "use": True,
            "batch_size": 1000,
            "partition": "trivial",
            "nesterov": {"use": False, "damping_int": 1.0},
        }
    )
    cfg = from_dictconfig_logs_hg(node)
    assert cfg.use is True
    assert cfg.batch_size == 1000
    assert cfg.damping == HgCfg().damping  # inherited default, not redeclared


def test_a_setting_the_logger_cannot_read_is_refused(f64):
    """The diagnostic is a NewtonSummaryFB, so the node is validated against it: a
    field only the stochastic variants read is a dead setting here."""
    from omegaconf import OmegaConf

    node = OmegaConf.create({"use": True, "partition": "trivial", "uniform_avg": {"period": 3}})
    with pytest.raises(ValueError, match="silently ignored"):
        from_dictconfig_logs_hg(node)


def test_the_shipped_logs_hg_node_validates(f64):
    """configs/config.yaml must compose. It carried `partition_arg: 0` alongside
    `partition: trivial` (rejected by HgCfg.__post_init__) and a `test_float` key
    in no schema at all (rejected by the merge)."""
    import pathlib

    from omegaconf import OmegaConf

    root = pathlib.Path(__file__).resolve().parents[1]
    node = OmegaConf.load(root / "configs" / "config.yaml").logs_hg
    cfg = from_dictconfig_logs_hg(node)
    assert isinstance(cfg, LogsCfg)
