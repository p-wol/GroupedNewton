#!/bin/bash
# slurm/submit.sh -- single entry point for both Slurm submission and local execution.
#
#   ./slurm/submit.sh cluster=jz_v100_dev expe_series=foo optimizer.lr=1e-3,1e-4
#   GRNEWT_LOCAL=1 ./slurm/submit.sh machine=none system.device=-2 ...
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

# --- .env ---------------------------------------------------------------------------
# Read line by line rather than `export $(sed ... | xargs)`: xargs word-splits on every
# space, so a path containing one silently became two exports and `set -e` killed the
# script with a message about an identifier rather than about configuration.
#
# Accepted on the left-hand side, because both conventions are in use in the wild and
# the old xargs pipeline tolerated the first one by accident (`export` became a separate
# word and `export export` is a harmless no-op):
#     KEY=value
#     export KEY=value
# Surrounding single or double quotes on the value are stripped, as a shell would.
# `env.example` documents the bare form; that is the one to write in new files.
if [ ! -f .env ]; then
    echo "submit.sh: no .env at $PWD; copy env.example and fill it in." >&2
    exit 1
fi
env_lineno=0
# `|| [ -n "$env_key" ]` so that a last line without a trailing newline is not dropped.
while IFS='=' read -r env_key env_val || [ -n "$env_key" ]; do
    env_lineno=$((env_lineno + 1))
    env_key="${env_key%$'\r'}"; env_val="${env_val%$'\r'}"        # CRLF
    env_key="${env_key#"${env_key%%[![:space:]]*}"}"               # ltrim
    env_key="${env_key%"${env_key##*[![:space:]]}"}"               # rtrim
    case "$env_key" in "" | \#*) continue ;; esac
    env_key="${env_key#export }"                                   # `export KEY=value`
    env_key="${env_key#"${env_key%%[![:space:]]*}"}"
    case "$env_val" in
        \"*\") env_val="${env_val#\"}"; env_val="${env_val%\"}" ;;
        \'*\') env_val="${env_val#\'}"; env_val="${env_val%\'}" ;;
    esac
    case "$env_key" in
        [A-Za-z_]*) : ;;
        *) echo "submit.sh: .env line $env_lineno: '$env_key' is not a variable name" >&2
           exit 1 ;;
    esac
    case "$env_key" in
        *[!A-Za-z0-9_]*)
           echo "submit.sh: .env line $env_lineno: '$env_key' is not a variable name" >&2
           exit 1 ;;
    esac
    export "$env_key=$env_val"
done < .env
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
    : "${GRNEWT_PROJECT:?set GRNEWT_PROJECT to your 3-letter IDRIS project code}"
    export GRNEWT_PROJECT
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
