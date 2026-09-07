"""The schema is the contract. These tests keep it from drifting.

They are the reason the design is maintainable rather than merely typed: adding a
field with no documentation, or a field the YAML does not know about, or a field no
optimizer reads, fails here instead of surfacing three months later as a run whose
config file does not describe what it did.
"""

import dataclasses

import pytest
import torch

from grnewt import ParamStructure  # noqa: F401
from grnewt import partition as build_partition
from grnewt.config import (
    ALL_NS,
    HgCfg,
    Partition,
    check_consumed,
    from_dictconfig,
    markdown_table,
    migrate,
)

omegaconf = pytest.importorskip("omegaconf")
OmegaConf = omegaconf.OmegaConf


def _walk(cfg, prefix=""):
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        path = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(v):
            yield from _walk(v, prefix=f"{path}.")
        else:
            yield path, v, f


# ---------------------------------------------------------------- documentation


def test_every_field_is_documented_and_attributed():
    """One line of help and a non-empty `used_by` per field. No exceptions."""
    missing = [
        p
        for p, _, f in _walk(HgCfg())
        if not f.metadata.get("help") or not f.metadata.get("used_by")
    ]
    assert not missing, f"fields without help/used_by: {missing}"


def test_used_by_names_are_known():
    for p, _, f in _walk(HgCfg()):
        unknown = f.metadata["used_by"] - ALL_NS
        assert not unknown, f"{p}: unknown optimizer name(s) {unknown}"


def test_markdown_table_covers_every_field():
    table = markdown_table()
    for p, _, _ in _walk(HgCfg()):
        assert f"`{p}`" in table, f"{p} missing from the generated documentation"


# ---------------------------------------------------------------- YAML drift


def test_schema_accepts_the_shipped_config_and_defaults_agree():
    """configs/config.yaml must compose against the schema, with identical defaults.

    Catches the two directions of drift: a YAML key the schema does not know, and a
    default changed on one side only -- which would silently invalidate the reference
    LeNet/VGG numbers.
    """
    pytest.importorskip("hydra")
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs"
    if not cfg_dir.is_dir():
        pytest.skip("configs/ not present in this checkout")

    import os

    os.environ.setdefault("GRNEWT_PROJECT", "test")
    os.environ.setdefault("GRNEWT_DATASETS", "/tmp/ds")
    os.environ.setdefault("GRNEWT_RESULTS", "/tmp/res")

    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        composed = compose(config_name="config", overrides=["run_id=test"])
    cfg = from_dictconfig(migrate(composed.optimizer.hg))

    default = HgCfg()
    differing = [
        p for (p, v, _), (_, d, _) in zip(_walk(cfg), _walk(default), strict=False) if v != d
    ]
    assert not differing, (
        "configs/config.yaml and grnewt/config.py disagree on the default of: "
        f"{differing}. Pick one source of truth (the dataclass) and make the YAML "
        "carry only deviations."
    )


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "node, msg",
    [
        ({"period_hg": "abc"}, "converted to Integer"),
        ({"nesterov": {"use": "maybe"}}, "not a valid bool"),
        ({"nesterov": {"dampng_int": 1.0}}, "not in 'NesterovCfg'"),
        ({"uniform_avg": {"period": 0}}, "must be >= 1"),
        ({"partition": "blocks"}, "requires partition_arg"),
        ({"partition": "canonical", "partition_arg": 4}, "meaningless"),
        ({"noregul": True, "nesterov": {"use": True}}, "mutually exclusive"),
        ({"nesterov": {"threshold_D_sing": 1e-5}}, None),  # legal: relative, in [0,1)
    ],
)
def test_invalid_nodes_are_rejected(node, msg):
    if msg is None:
        from_dictconfig(OmegaConf.create(node))
        return
    with pytest.raises(Exception) as e:
        from_dictconfig(OmegaConf.create(node))
    assert msg in str(e.value)


# ---------------------------------------------------------------- migration


@pytest.mark.parametrize(
    "legacy, kind, arg, string",
    [
        ("canonical", Partition.canonical, None, None),
        ("blocks-4", Partition.blocks, 4, None),
        ("alternate-2", Partition.alternate, 2, None),
        ("vgg-features", Partition.vgg, None, "features"),
    ],
)
def test_legacy_partition_strings_still_work(legacy, kind, arg, string):
    cfg = from_dictconfig(migrate(OmegaConf.create({"partition": legacy})))
    assert (cfg.partition, cfg.partition_arg, cfg.partition_str) == (kind, arg, string)


@pytest.mark.parametrize(
    "legacy, use, final, epoch",
    [("None", False, 1.0, 0), ("0.1-50", True, 0.1, 50)],
)
def test_legacy_damping_schedule_still_works(legacy, use, final, epoch):
    cfg = from_dictconfig(migrate(OmegaConf.create({"damping_schedule": legacy})))
    d = cfg.damping_schedule
    assert (d.use, d.final, d.epoch) == (use, final, epoch)


# ---------------------------------------------------------------- ignored settings


def test_default_config_is_consumed_by_every_optimizer():
    """An all-defaults config must never trip the detector, whatever the optimizer."""
    for name in sorted(ALL_NS):
        assert check_consumed(HgCfg(), name) == []


# ---------------------------------------------------------------- smoke


def test_optimizer_builds_from_the_default_config_and_steps():
    from grnewt import NewtonSummaryUniformAvg
    from grnewt.optimizers import SGDUpdate

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
    x, y = torch.randn(16, 6), torch.randn(16, 3)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=8, shuffle=False
    )
    loss_fn = torch.nn.MSELoss()

    def full_loss(a, b):
        return loss_fn(model(a), b)

    pgroups, _ = build_partition.canonical(model)
    cfg = from_dictconfig(
        OmegaConf.create({"nesterov": {"use": True}}),
        optimizer_name="NewtonSummaryUniformAvg",
    )
    opt = NewtonSummaryUniformAvg(
        pgroups,
        full_loss,
        dl,
        SGDUpdate(model.parameters(), lr=1, momentum=0.9),
        loader_pre_hook=lambda a, b: (a, b),
        cfg=cfg,
    )
    before = [p.detach().clone() for p in model.parameters()]
    for xb, yb in dl:
        opt.zero_grad()
        full_loss(xb, yb).backward()
        opt.step()
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=False))
    assert all(torch.isfinite(p).all() for p in after)
