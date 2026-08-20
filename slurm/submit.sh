#!/bin/bash
# slurm/submit.sh -- single entry point for both Slurm submission and local execution.
#
#   ./slurm/submit.sh cluster=jz_v100_dev expe_series=foo optimizer.lr=1e-3,1e-4
#   GRNEWT_LOCAL=1 ./slurm/submit.sh paths=local cluster=none system.device=-2 ...
#
# Mode:
#   slurm  -- default when `srun` is on PATH. Composes, then hands the sweep to Hydra's
#             submitit launcher (--multirun).
#   local  -- when `srun` is absent, or when GRNEWT_LOCAL=1. Enumerates the sweep with
#             slurm/expand_sweeps.py and runs the configurations sequentially in this
#             process. No submitit: hydra-submitit-launcher's SlurmExecutor raises
#             'Could not detect "srun"' at construction time, so --multirun is simply
#             not usable off-cluster.
#
# Why `run_id` is fixed here and passed as an override: Hydra's `${now:}` resolver is
# evaluated per process, and with submitit the job re-composes the config in a fresh
# process on a compute node. A CLI override is pickled with the job and is therefore the
# same string on both sides by construction.
#
# Everything after the script name is forwarded to Hydra verbatim.

set -euo pipefail
cd "$(dirname "$0")/.."

export $(sed '/^#/d; /^[[:space:]]*$/d' .env | xargs)
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1

# --- mode ---------------------------------------------------------------------------
if [ "${GRNEWT_LOCAL:-0}" = "1" ]; then
    MODE=local
elif command -v srun >/dev/null 2>&1; then
    MODE=slurm
else
    MODE=local
    echo "note: srun not found on PATH -> local mode (set GRNEWT_LOCAL=1 to silence)"
fi

if [ "$MODE" = "slurm" ]; then
    # 3-letter IDRIS project code, without the @v100 suffix. Set it in ~/.bashrc.
    : "${JZ_PROJECT:?set JZ_PROJECT to your 3-letter IDRIS project code}"
    export JZ_PROJECT
    # The environment used to *submit* must be the one the job will use: submitit
    # hardcodes sys.executable of the submitting process into the sbatch script.
    module purge
    module load pytorch-gpu/py3/2.8.0
fi

# --- dry composition, nothing run or submitted ---------------------------------------
# One check, on the exact argument list used below. NOT `main_hydra.py --cfg job`:
# that is a single-run composition and rejects `optimizer.lr=a,b,c` as ambiguous.
# check_config.py collapses sweeps before composing and resolves job config + launcher.
echo "=== checking configuration (mode: $MODE) ==="
python slurm/check_config.py "--mode=$MODE" "run_id=${RUN_ID}" "$@"

# --- run ------------------------------------------------------------------------------
if [ "$MODE" = "slurm" ]; then
    echo "=== submitting (run_id=${RUN_ID}) ==="
    python main_hydra.py --multirun "run_id=${RUN_ID}" "$@"
    echo "=== submitted. Logs: <results>/<expe_series>/${RUN_ID}/.submitit/ ==="
else
    COMBOS="$(python slurm/expand_sweeps.py "$@")"
    N="$(printf '%s\n' "$COMBOS" | grep -c . || true)"
    echo "=== running ${N} configuration(s) sequentially (run_id=${RUN_ID}) ==="
    i=0
    while IFS= read -r combo; do
        [ -z "$combo" ] && continue
        echo "--- [$i/${N}] $combo ---"
        # hydra.run.dir is given in interpolation form so that it resolves exactly as
        # hydra.sweep.dir would, with one subdirectory per configuration.
        eval python main_hydra.py "run_id=${RUN_ID}" "$combo" \
            "'hydra.run.dir=\${paths.results}/\${expe_series}/\${run_id}/$i'"
        i=$((i+1))
    done <<< "$COMBOS"
    echo "=== done. Results: <results>/<expe_series>/${RUN_ID}/ ==="
fi
