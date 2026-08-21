#!/usr/bin/env python
"""tools/compare_runs.py -- compare epoch times across the runs of a --multirun sweep.

The companion of a tuning sweep. It does not launch anything and knows nothing about the
experiment: it reads the artefacts the trainer already writes, so the "tuning script" is
just the ordinary launch script plus a Hydra override.

    ./slurm/run_lenet_cifar.sh expe_series=tune dataset.num_workers=2,5,9 \\
        optimizer.epochs=4
    python tools/compare_runs.py $WORK/.../GroupedNewton_Results/tune/<run_id>

For each job directory it reads:
  * .hydra/overrides.yaml  -> the overrides that distinguish this run
  * **/metrics.json        -> one Python-repr dict per line, with 'epoch' and 'time'
                              (cumulative seconds since the start of training)

Epoch duration is the successive difference of 'time'. Epoch 0 is dropped: it carries the
CUDA context, cuDNN algorithm selection, the first read of the dataset and, when workers
are used, the spawning of the worker processes.

Reported per run: median, min and max epoch duration. The median is the right statistic
here -- one epoch delayed by a Lustre write should not move the number.
"""

from __future__ import annotations

import ast
import statistics
import sys
from pathlib import Path


def read_metrics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(ast.literal_eval(line))
        except (ValueError, SyntaxError):
            pass
    return rows


def epoch_durations(rows: list[dict]) -> list[float]:
    rows = sorted((r for r in rows if "time" in r and "epoch" in r),
                  key=lambda r: r["epoch"])
    return [b["time"] - a["time"] for a, b in zip(rows, rows[1:], strict=False)]


def read_overrides(job_dir: Path) -> list[str]:
    f = job_dir / ".hydra" / "overrides.yaml"
    if not f.exists():
        return []
    try:
        import yaml

        return list(yaml.safe_load(f.read_text()) or [])
    except Exception:
        return []


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    root = Path(argv[0])
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2

    runs = []
    for m in sorted(root.glob("**/metrics.json")):
        job_dir = m.parent
        while job_dir != root and not (job_dir / ".hydra").is_dir():
            job_dir = job_dir.parent
        durations = epoch_durations(read_metrics(m))[1:]  # drop epoch 0->1
        runs.append({"dir": job_dir, "overrides": read_overrides(job_dir),
                     "durations": durations,
                     "peak": max((r.get("memory_peak", 0)
                                  for r in read_metrics(m)), default=0)})

    if not runs:
        print(f"no metrics.json found under {root}")
        return 1

    # Keep only the overrides that actually differ between runs: that is the axis swept.
    keys = {o.split("=")[0] for r in runs for o in r["overrides"]}
    varying = sorted(k for k in keys
                     if len({next((o for o in r["overrides"]
                                   if o.startswith(k + "=")), None)
                             for r in runs}) > 1)

    print(f"{len(runs)} run(s) under {root}")
    print(f"axis: {', '.join(varying) if varying else '(nothing varies)'}")
    print(f"{'run':<40} {'n':>3} {'median':>9} {'min':>9} {'max':>9} {'peak MiB':>9}")
    rows = []
    for r in runs:
        label = " ".join(o for o in r["overrides"]
                         if any(o.startswith(k + "=") for k in varying)) or r["dir"].name
        d = r["durations"]
        if not d:
            print(f"{label:<40} {0:>3}   (fewer than 3 epochs logged)")
            continue
        rows.append((statistics.median(d), label, len(d), min(d), max(d),
                     r["peak"] / 2**20))
    rows.sort()   # ascending median: rows[0] is the fastest configuration
    for med, label, n, lo, hi, peak in rows:
        print(f"{label:<40} {n:>3} {med:>8.2f}s {lo:>8.2f}s {hi:>8.2f}s {peak:>9.0f}")

    if len(rows) > 1:
        best, worst = rows[0], rows[-1]
        print(f"\nbest: {best[1]}  ({best[0]:.2f} s/epoch, "
              f"{worst[0] / best[0]:.2f}x faster than {worst[1]})")
        print("Read the plateau, not the minimum: pick the smallest setting whose median "
              "is within a few percent of the best, and check that the loss curves match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
