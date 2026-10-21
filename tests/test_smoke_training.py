"""End-to-end: a real training run must actually reduce the loss.

Referenced by the `smoke-train` CI job, which invokes
`pytest -q -m slow tests/test_smoke_training.py`. Kept CPU-only and small enough
to finish in well under the 120 s pytest timeout.

These are the cheapest possible guards against the two failure classes that unit
tests structurally cannot see:
  * the seam between the updater and ParamStructure (a permuted `direction`
    broadcasts silently rather than raising -- see test_regressions_20260821);
  * the running averages inside NewtonSummaryUniformAvg, which only misbehave
    across several Hg-update periods.
"""

import pytest
import torch

from grnewt import NewtonSummaryMovexpAvg, NewtonSummaryUniformAvg, optimizers
from grnewt import partition as build_partition
from grnewt.config import HgCfg, NesterovCfg, UniformAvgCfg

pytestmark = pytest.mark.slow


def _problem(seed=0, n=256, d_in=8, d_out=3, width=16):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, width),
        torch.nn.Tanh(),
        torch.nn.Linear(width, width),
        torch.nn.Tanh(),
        torch.nn.Linear(width, d_out),
    )
    x = torch.randn(n, d_in)
    w = torch.randn(d_in, d_out)
    y = (x @ w).argmax(dim=1)
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    hg_loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    def epoch_loss():
        with torch.no_grad():
            return float(sum(full_loss(a, b) * a.size(0) for a, b in loader) / len(ds))

    return model, loader, hg_loader, full_loss, epoch_loss


PARTITIONS = {
    "canonical": build_partition.canonical,
    "wb": build_partition.wb,
    "blocks-2": lambda m: build_partition.blocks(m, 2),
}


@pytest.mark.parametrize("partition", sorted(PARTITIONS))
def test_newton_summary_reduces_the_loss(partition):
    model, loader, hg_loader, full_loss, epoch_loss = _problem()
    pgroups, _ = PARTITIONS[partition](model)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    cfg = HgCfg(
        damping=0.1,
        period_hg=4,
        remove_negative=True,
        nesterov=NesterovCfg(use=True, damping_int=1.0),
    )
    opt = NewtonSummaryMovexpAvg(
        pgroups,
        full_loss,
        hg_loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )

    before = epoch_loss()
    for _ in range(6):
        for a, b in loader:
            updater.zero_grad()
            full_loss(a, b).backward()
            opt.step()
    after = epoch_loss()

    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert after < before, f"partition={partition}: loss {before:.4f} -> {after:.4f}"


def test_uniform_avg_reduces_the_loss_across_several_periods():
    model, loader, hg_loader, full_loss, epoch_loss = _problem(seed=1)
    pgroups, _ = build_partition.canonical(model)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    cfg = HgCfg(
        damping=0.1,
        period_hg=4,
        remove_negative=True,
        nesterov=NesterovCfg(use=True, damping_int=1.0),
        uniform_avg=UniformAvgCfg(period=3, warmup=3),
    )
    opt = NewtonSummaryUniformAvg(
        pgroups,
        full_loss,
        hg_loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )

    before = epoch_loss()
    for _ in range(10):
        for a, b in loader:
            updater.zero_grad()
            full_loss(a, b).backward()
            opt.step()
    after = epoch_loss()

    assert all(torch.isfinite(p).all() for p in model.parameters())
    # the averages must be genuine averages, not zeros: a zeroed H_use makes
    # nesterov_lrs take the trivial_g0 branch and the run stalls at `before`
    assert opt.logs["H"], "no (H, g, D) was ever recorded"
    assert float(opt.logs["H"][-1].abs().max()) > 0.0
    assert after < before, f"loss {before:.4f} -> {after:.4f}"


def test_solver_failures_are_reported_not_silent():
    """`nesterov_lrs` returning None must never be swallowed: the optimizer logs
    it under `nesterov.found`, which is what the run's post-mortem reads."""
    model, loader, hg_loader, full_loss, _ = _problem(seed=2)
    pgroups, _ = build_partition.canonical(model)
    updater = optimizers.SGDUpdate(model.parameters(), lr=1, momentum=0.9)
    cfg = HgCfg(period_hg=1, nesterov=NesterovCfg(use=True, damping_int=1.0))
    opt = NewtonSummaryMovexpAvg(
        pgroups,
        full_loss,
        hg_loader,
        updater,
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )
    for a, b in loader:
        updater.zero_grad()
        full_loss(a, b).backward()
        opt.step()

    assert "nesterov.found" in opt.logs
    assert len(opt.logs["nesterov.found"]) == len(opt.logs["H"])
    # every accepted step must carry an r above the pole
    for found, r in zip(opt.logs["nesterov.found"], opt.logs["nesterov.r"], strict=False):
        if found:
            assert torch.isfinite(torch.as_tensor(r)).all()


def test_training_entry_point_runs_end_to_end(tmp_path):
    """`main_hydra.py` -> `training_hydra.Trainer` must complete a (tiny) run, with the
    full-batch `logs_hg` path on. Guards C27 (training_hydra imported a name grnewt no
    longer defined) and C37 (the optimizer was built before `loader_pre_hook` existed
    for ToyRegression / None). Success is the `DONE` sentinel, not the exit code alone:
    under Hydra a failed task can still exit 0 (STATE V31)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("hydra")
    pytest.importorskip("torchvision")
    repo = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ, GRNEWT_DATASETS=str(tmp_path / "ds"), GRNEWT_RESULTS=str(tmp_path / "res")
    )
    cmd = [
        sys.executable,
        str(repo / "main_hydra.py"),
        "machine=none",
        "run_id=smoke",
        "system.device=-2",
        "dataset.name=ToyRegression",
        "model.name=Perceptron",
        "model.args='4-8-1'",
        "dataset.train_size=200",
        "dataset.valid_size=50",
        "dataset.test_size=50",
        "dataset.batch_size=20",
        "dataset.teacher.args='4-1'",
        "logs_hg.use=True",
        "logs_hg.batch_size=30",
        "optimizer.epochs=2",
        "optimizer.name=NewtonSummaryUniformAvg",
        "optimizer.hg.period_hg=2",
        "optimizer.hg.uniform_avg.period=2",
        "optimizer.hg.nesterov.use=True",
        "optimizer.hg.remove_negative=True",
    ]
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=110)
    out = tmp_path / "res" / "tests" / "smoke"
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert (out / "DONE").exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    assert sorted(p.name for p in (out / "artifacts").glob("Hg_logs_ext.*.pkl")), (
        "logs_hg not written"
    )
