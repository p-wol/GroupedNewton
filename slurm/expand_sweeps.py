#!/usr/bin/env python
"""slurm/expand_sweeps.py -- expand Hydra sweep overrides into one line per run.

Used by slurm/submit.sh in local mode: with no Slurm, there is no sweeper, so the sweep
has to be enumerated and run sequentially. Hydra's own override parser does the parsing,
so `a=1,2`, `a=range(1,4)` and `a=choice(x,y)` are all handled the same way they would be
by the BasicSweeper.

    $ python slurm/expand_sweeps.py optimizer.lr=1e-3,1e-4 seed=1
    optimizer.lr=1e-3 seed=1
    optimizer.lr=1e-4 seed=1

Each line is shell-quoted and meant to be consumed with `eval`.
"""

from __future__ import annotations

import itertools
import shlex
import sys


def expand(overrides: list[str]) -> list[list[str]]:
    from hydra.core.override_parser.overrides_parser import OverridesParser

    parsed = OverridesParser.create().parse_overrides(overrides=overrides)
    choices: list[list[str]] = []
    for raw, ov in zip(overrides, parsed):
        if ov.is_sweep_override():
            key = ov.get_key_element()
            choices.append([f"{key}={v}" for v in ov.sweep_string_iterator()])
        else:
            choices.append([raw])
    return [list(c) for c in itertools.product(*choices)]


def main(argv: list[str]) -> int:
    if not argv:
        return 0
    try:
        combos = expand(argv)
    except Exception as exc:
        print(f"expand_sweeps: cannot parse overrides: {exc!r}", file=sys.stderr)
        return 2
    for combo in combos:
        print(" ".join(shlex.quote(c) for c in combo))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
