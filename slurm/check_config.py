#!/usr/bin/env python
"""slurm/check_config.py -- compose the config and resolve the launcher node.

Why this exists instead of `python main_hydra.py --cfg hydra --resolve`:
`--cfg hydra` returns a *sanitized* hydra config, from which Hydra deletes every
top-level key other than `hydra`. `paths` is therefore gone, and `--resolve` then fails
on `${paths.results}` -- an artefact of the inspection command, not a defect of the
config. Composing with `return_hydra_config=True` keeps both nodes in the same root, so
the interpolations resolve exactly as they will at run time.

Only `hydra.launcher` and the output directories are resolved. `hydra.sweep.subdir` is
deliberately left alone: it interpolates `${hydra.job.num}`, which is MISSING outside an
actual job and would raise here for no reason.

Usage (from the repo root):
    python slurm/check_config.py run_id=check cluster=jz_v100_dev expe_series=smoke
Exit code 0 = the submission would compose. Nothing is submitted.
"""

from __future__ import annotations

import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main(overrides: list[str]) -> int:
    config_dir = os.path.abspath("configs")
    if not os.path.isdir(config_dir):
        print(f"FAIL  {config_dir} does not exist (run this from the repo root)")
        return 2

    with initialize_config_dir(version_base="1.3", config_dir=config_dir,
                               job_name="check_config"):
        cfg = compose(config_name="config", overrides=overrides,
                      return_hydra_config=True)

        print("--- paths ---")
        for k in ("work", "scratch", "datasets", "results"):
            print(f"  paths.{k:9s} = {cfg.paths[k]}")

        print("--- output directories ---")
        print(f"  hydra.run.dir   = {cfg.hydra.run.dir}")
        print(f"  hydra.sweep.dir = {cfg.hydra.sweep.dir}")

        print("--- hydra.launcher (resolved) ---")
        launcher = cfg.hydra.launcher
        OmegaConf.resolve(launcher)
        print(OmegaConf.to_yaml(launcher))

        # Assertions on the fields whose absence produces an opaque Slurm failure.
        problems = []
        if not launcher.get("account"):
            problems.append("account is empty: mandatory on Jean Zay")
        if launcher.get("gpus_per_node") is not None and launcher.get("gres"):
            problems.append("gpus_per_node AND gres are both set: two GPU directives")
        if not launcher.get("gres") and launcher.get("gpus_per_node") is None:
            problems.append("no GPU requested (gres and gpus_per_node both empty)")
        if "submitit" not in str(launcher.get("_target_", "")):
            problems.append(f"_target_ is {launcher.get('_target_')!r}, not a submitit "
                            f"launcher: the 'override hydra/launcher' line is not applied")
        for p in problems:
            print(f"FAIL  {p}")
        if problems:
            return 1
        print("PASS  launcher composes and resolves")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
