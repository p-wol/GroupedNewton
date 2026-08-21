#!/usr/bin/env python
"""slurm/check_config.py -- compose and resolve the config, without running anything.

Two modes, because the invariants differ:

  slurm  the config will be handed to submitit. `account`, `gres` and a submitit
         `_target_` must be present: their absence produces an opaque Slurm failure.
  local  the config will be run in this process. `hydra.launcher` is composed but never
         instantiated (Hydra only instantiates a launcher in --multirun), so asserting
         anything about it is meaningless. What matters instead is that CUDA agrees with
         `system.device`.

Mode is auto-detected from the selected `machine` option (`machine=none` -> local) and
can be forced with --mode=slurm|local.

Design notes:
  * `--cfg hydra` is not usable for this: Hydra strips every non-hydra top-level key from
    that view, so `${paths.results}` becomes unresolvable. `return_hydra_config=True`
    keeps both nodes in one root, which is how they resolve at run time.
  * sweep overrides (`optimizer.lr=a,b,c`) are collapsed to their first value: any
    single-run composition rejects them as ambiguous. The check therefore accepts the
    exact argument list that will later be passed to --multirun, but validates only the
    first configuration of the sweep.
  * `hydra.sweep.subdir` is left unresolved: it interpolates `${hydra.job.num}`, MISSING
    outside an actual job.

Usage, from the repo root:
    python slurm/check_config.py run_id=check machine=jz_v100_dev optimizer.lr=1e-3,1e-4
    python slurm/check_config.py --mode=local run_id=check machine=none
Exit code 0 = the run would compose and resolve.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_NAIVE = re.compile(r"^([^=]+)=([^,\[\]()]+),.*$")

# torchvision layouts, from src/grnewt/datasets.py (download=False everywhere).
_EXPECTED_SUBDIR = {
    "MNIST": ("MNIST/raw", True),
    "CIFAR10": ("cifar-10-batches-py", True),
    "ImageNet": ("train", False),  # layouts vary; warn only
}


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def ok(self, msg: str) -> None:
        print(f"  PASS  {msg}")

    def warn(self, msg: str) -> None:
        print(f"  WARN  {msg}")

    def bad(self, msg: str) -> None:
        print(f"  FAIL  {msg}")
        self.failed += 1


def collapse_sweeps(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Replace each sweep override by its first value. Returns (overrides, notes)."""
    out: list[str] = []
    notes: list[str] = []
    try:
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parsed = OverridesParser.create().parse_overrides(overrides=overrides)
    except Exception:
        parsed = None

    if parsed is not None:
        for raw, ov in zip(overrides, parsed, strict=False):
            try:
                if ov.is_sweep_override():
                    first = next(iter(ov.sweep_string_iterator()))
                    collapsed = f"{ov.get_key_element()}={first}"
                    notes.append(f"{raw}  ->  {collapsed}")
                    out.append(collapsed)
                    continue
            except Exception:
                pass
            out.append(raw)
        return out, notes

    for raw in overrides:  # fallback if the parser API moved
        m = _NAIVE.match(raw)
        if m:
            collapsed = f"{m.group(1)}={m.group(2)}"
            notes.append(f"{raw}  ->  {collapsed}  (naive split)")
            out.append(collapsed)
        else:
            out.append(raw)
    return out, notes


def detect_mode(cfg) -> str:
    """Must not resolve hydra.launcher: on a machine with no GRNEWT_PROJECT, resolving
    `account: ${oc.env:GRNEWT_PROJECT}@v100` raises, and that machine is precisely the
    one where the answer is 'local'."""
    return cfg.launch_mode
    """
    try:
        return "local" if str(cfg.hydra.runtime.choices.machine) in ("none", "laptop") else "slurm"
    except Exception:
        pass
    try:
        launcher = cfg.hydra.launcher
        if not launcher.get("account") and not launcher.get("gres"):
            return "local"
        return "slurm"
    except Exception:
        return "local"  # unresolvable launcher: submission is impossible anyway
    """


def check_paths(job_cfg, rep: Report) -> None:
    print("--- paths ---")
    for k in ("datasets", "results"):
        print(f"  paths.{k:9s} = {job_cfg.paths[k]}")

    results = Path(str(job_cfg.paths.results))
    anc = results
    while not anc.exists() and anc != anc.parent:
        anc = anc.parent
    if os.access(anc, os.W_OK):
        rep.ok(f"results writable (nearest existing ancestor: {anc})")
    else:
        rep.bad(f"{anc} is not writable: no output directory, and no log either")


def check_dataset(job_cfg, rep: Report) -> None:
    name = str(job_cfg.dataset.name)
    root = Path(str(job_cfg.dataset.path))
    print(f"--- dataset: {name} under {root} ---")
    if name not in _EXPECTED_SUBDIR:
        rep.ok(f"{name}: no on-disk data expected")
        return
    sub, hard = _EXPECTED_SUBDIR[name]
    target = root / sub
    if target.exists():
        rep.ok(f"{target} found")
    elif hard:
        rep.bad(f"{target} not found. download=False in datasets.py and compute nodes "
                f"have no internet: this fails at epoch 0, after the allocation is "
                f"granted. Fix paths.datasets or dataset.path.")
    else:
        rep.warn(f"{target} not found; ImageNet layouts vary, check by hand")


def check_device(job_cfg, rep: Report) -> None:
    dev = int(job_cfg.system.device)
    if dev < -2:
        rep.bad(f"system.device={dev} is out of range. NB: assign_device() builds a "
                f"ValueError without raising it, so this returns an int silently.")
        return
    if dev == -2:
        rep.ok("system.device=-2 -> CPU, no GPU needed")
        return
    try:
        import torch
    except Exception as exc:
        rep.warn(f"torch not importable here ({exc!r}); GPU availability not checked")
        return
    if torch.cuda.is_available():
        rep.ok(f"system.device={dev} and CUDA available "
               f"({torch.cuda.device_count()} device(s))")
    else:
        rep.bad(f"system.device={dev} requires CUDA, unavailable here. Use "
                f"system.device=-2 for a CPU run. (assign_device() would silently fall "
                f"back to CPU; main_hydra.py turns that into an error.)")


def check_launcher(launcher, rep: Report) -> None:
    print("--- hydra.launcher (resolved) ---")
    print(OmegaConf.to_yaml(launcher))
    if not launcher.get("account"):
        rep.bad("account is empty: mandatory on Jean Zay")
    else:
        rep.ok(f"account = {launcher.get('account')}")
    if launcher.get("gpus_per_node") is not None and launcher.get("gres"):
        rep.bad("gpus_per_node AND gres are both set: two GPU directives")
    if not launcher.get("gres") and launcher.get("gpus_per_node") is None:
        rep.bad("no GPU requested (gres and gpus_per_node both empty)")
    if "submitit" not in str(launcher.get("_target_", "")):
        rep.bad(f"_target_ is {launcher.get('_target_')!r}, not a submitit launcher: "
                f"'override hydra/launcher' is not applied")
    else:
        rep.ok(f"_target_ = {launcher.get('_target_')}")


def check_optimizer_schema(cfg, rep: Report) -> None:
    """Validate `optimizer.hg` against the typed schema, here, on the login node.

    This is the whole point of grnewt/config.py: a type error, an out-of-range value,
    or a setting the selected optimizer does not read must cost 200 ms here rather
    than a Slurm allocation. Under the submitit launcher the task function runs on the
    compute node, so validating inside the trainer is too late.
    """
    print("--- optimizer.hg schema ---")
    name = cfg.optimizer.name
    if not str(name).startswith("NewtonSummary"):
        rep.ok(f"optimizer.name={name}: no hg schema to check")
        return
    try:
        from grnewt.config import from_dictconfig, migrate
    except ImportError as exc:
        rep.warn(f"grnewt not importable, hg schema unchecked ({exc})")
        return
    try:
        hg = from_dictconfig(migrate(cfg.optimizer.hg), optimizer_name=name)
    except Exception as exc:
        lines = str(exc).splitlines() or [repr(exc)]
        rep.bad(lines[0])
        for line in lines[1:]:
            print(f"        {line}")
        return
    rep.ok(f"optimizer.hg validates for {name} (partition={hg.partition.value}, "
           f"nesterov.use={hg.nesterov.use})")


def main(argv: list[str]) -> int:
    forced_mode = None
    overrides = []
    for a in argv:
        if a.startswith("--mode="):
            forced_mode = a.split("=", 1)[1]
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            overrides.append(a)
    if forced_mode not in (None, "slurm", "local"):
        print(f"FAIL  unknown --mode={forced_mode} (expected slurm or local)")
        return 2

    config_dir = os.path.abspath("configs")
    if not os.path.isdir(config_dir):
        print(f"FAIL  {config_dir} does not exist (run this from the repo root)")
        return 2

    overrides, notes = collapse_sweeps(overrides)
    if notes:
        print("--- sweeps collapsed for this check only ---")
        for n in notes:
            print(f"  {n}")

    rep = Report()
    with initialize_config_dir(version_base="1.3", config_dir=config_dir,
                               job_name="check_config"):
        cfg = compose(config_name="config", overrides=overrides,
                      return_hydra_config=True)

        # the experiment config must resolve on its own
        job_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        del job_cfg["hydra"]
        OmegaConf.resolve(job_cfg)

        mode = forced_mode or detect_mode(cfg)
        print(f"=== mode: {mode} "
              f"({'forced' if forced_mode else 'auto-detected'}) ===")

        check_paths(job_cfg, rep)

        print("--- output directories ---")
        print(f"  hydra.run.dir   = {cfg.hydra.run.dir}")
        print(f"  hydra.sweep.dir = {cfg.hydra.sweep.dir}")

        check_dataset(job_cfg, rep)
        check_optimizer_schema(job_cfg, rep)

        if mode == "slurm":
            # the GPU of the compute node, not of this machine: nothing to check here
            launcher = cfg.hydra.launcher
            OmegaConf.resolve(launcher)   # only here: see detect_mode()
            check_launcher(launcher, rep)
        else:
            print("--- launcher ---")
            print("  not instantiated: Hydra only builds a launcher in --multirun")
            check_device(job_cfg, rep)

    if rep.failed:
        print(f"=== {rep.failed} failure(s) ===")
        return 1
    print("=== config composes and resolves ===")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
