#!/usr/bin/env python
"""slurm/check_config.py -- compose the config and resolve it, without submitting.

Two reasons this is a script rather than a `python main_hydra.py --cfg ...` invocation:

1. `--cfg hydra` returns a *sanitized* hydra config, from which Hydra deletes every
   top-level key other than `hydra`. `paths` is gone, and `--resolve` then fails on
   `${paths.results}`. Composing with `return_hydra_config=True` keeps both nodes in the
   same root, so interpolations resolve exactly as they will at run time.

2. Sweep overrides. `optimizer.lr=.0003,.0001,.00003` is only meaningful in multirun
   mode; any single-run composition rejects it as ambiguous. A pre-submission check must
   therefore collapse every sweep to its first value before composing. That is what
   `collapse_sweeps` does, and it is why this check can be run on the exact argument list
   that will later be passed to `--multirun`.

`hydra.sweep.subdir` is deliberately left unresolved: it interpolates
`${hydra.job.num}`, MISSING outside an actual job.

Usage, from the repo root:
    python slurm/check_config.py run_id=check cluster=jz_v100_dev optimizer.lr=1e-3,1e-4
Exit code 0 = the submission would compose and resolve. Nothing is submitted.
"""

from __future__ import annotations

import os
import re
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_NAIVE = re.compile(r"^([^=]+)=([^,\[\]()]+),.*$")


def collapse_sweeps(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Replace each sweep override by its first value. Returns (overrides, notes)."""
    out: list[str] = []
    notes: list[str] = []
    try:
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(overrides=overrides)
    except Exception:
        parsed = None

    if parsed is not None:
        for raw, ov in zip(overrides, parsed):
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

    # Fallback: the parser itself refused the list (should not happen).
    for raw in overrides:
        m = _NAIVE.match(raw)
        if m:
            collapsed = f"{m.group(1)}={m.group(2)}"
            notes.append(f"{raw}  ->  {collapsed}  (naive split)")
            out.append(collapsed)
        else:
            out.append(raw)
    return out, notes


def main(argv: list[str]) -> int:
    config_dir = os.path.abspath("configs")
    if not os.path.isdir(config_dir):
        print(f"FAIL  {config_dir} does not exist (run this from the repo root)")
        return 2

    overrides, notes = collapse_sweeps(argv)
    if notes:
        print("--- sweeps collapsed for this check only ---")
        for n in notes:
            print(f"  {n}")

    with initialize_config_dir(version_base="1.3", config_dir=config_dir,
                               job_name="check_config"):
        cfg = compose(config_name="config", overrides=overrides,
                      return_hydra_config=True)

        # 1. the experiment config must resolve on its own
        job_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        del job_cfg["hydra"]
        OmegaConf.resolve(job_cfg)

        print("--- paths ---")
        for k in ("work", "scratch", "datasets", "results"):
            print(f"  paths.{k:9s} = {job_cfg.paths[k]}")
        print(f"  dataset.path  = {job_cfg.dataset.path}")

        print("--- output directories ---")
        print(f"  hydra.run.dir   = {cfg.hydra.run.dir}")
        print(f"  hydra.sweep.dir = {cfg.hydra.sweep.dir}")

        # 2. the launcher must resolve
        print("--- hydra.launcher (resolved) ---")
        launcher = cfg.hydra.launcher
        OmegaConf.resolve(launcher)
        print(OmegaConf.to_yaml(launcher))

        problems = []
        if not launcher.get("account"):
            problems.append("account is empty: mandatory on Jean Zay")
        if launcher.get("gpus_per_node") is not None and launcher.get("gres"):
            problems.append("gpus_per_node AND gres are both set: two GPU directives")
        if not launcher.get("gres") and launcher.get("gpus_per_node") is None:
            problems.append("no GPU requested (gres and gpus_per_node both empty)")
        if "submitit" not in str(launcher.get("_target_", "")):
            problems.append(f"_target_ is {launcher.get('_target_')!r}, not a submitit "
                            f"launcher: 'override hydra/launcher' is not applied")
        for p in problems:
            print(f"FAIL  {p}")
        if problems:
            return 1
        print("PASS  job config and launcher compose and resolve")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
