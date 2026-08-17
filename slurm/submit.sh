#!/bin/bash
# slurm/submit.sh -- the single entry point for every Jean Zay submission.
#
#   ./slurm/submit.sh cluster=jz_v100_dev expe_series=foo optimizer.lr=1e-3,1e-4
#
# What it does, and why:
#   * fixes `run_id` ONCE, here, and passes it as a command-line override. Hydra's
#     `${now:...}` resolver is evaluated per process: with the submitit launcher the
#     job re-composes the config in a fresh process on a compute node, so a `${now:}`
#     inside hydra.sweep.dir can resolve to a different string than the one the launcher
#     used when it created the directory. A CLI override is pickled with the job and is
#     the same string on both sides by construction.
#   * exports GRNEWT_PROJECT so that `account: <proj>@v100` resolves. Slurm propagates
#     the environment (--export=ALL by default), so it is also defined inside the job.
#   * runs `--cfg job --resolve` first: composition errors are then reported here, on
#     the login node, with a full traceback, instead of inside a pickled job.
#
# Everything after the script name is forwarded to Hydra verbatim.

set -euo pipefail

# --- to be set once, e.g. in ~/.bashrc -------------------------------------------
# 3-letter project code (the one in `idrproj`), WITHOUT the @v100 suffix.
: "${GRNEWT_PROJECT:?set GRNEWT_PROJECT to your 3-letter IDRIS project code}"
export GRNEWT_PROJECT

# --- environment used to *submit*. It must be the same one the job will use, because
# submitit hardcodes sys.executable of the submitting process into the sbatch script.
module purge
module load pytorch-gpu/py3/2.4.0

cd "$(dirname "$0")/.."

RUN_ID="$(date -u +%Y%m%d-%H%M%S)"

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1

echo "=== dry composition, nothing submitted ==="
# One single check, on the exact argument list that will be swept below.
# NOT `python main_hydra.py --cfg job --resolve "$@"`: that is a single-run composition,
# which rejects `optimizer.lr=a,b,c` as ambiguous. check_config.py collapses sweeps to
# their first value before composing, and resolves both the job config and the launcher.
python slurm/check_config.py "run_id=${RUN_ID}" "$@"

echo "=== submitting (run_id=${RUN_ID}) ==="
python main_hydra.py --multirun "run_id=${RUN_ID}" "$@"

echo "=== submitted. Logs: <results>/<expe_series>/${RUN_ID}/.submitit/ ==="
