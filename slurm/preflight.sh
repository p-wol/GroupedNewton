#!/bin/bash
# slurm/preflight.sh (v2) -- run this on a Jean Zay login node before anything else.
# Each check corresponds to a failure mode whose Slurm-side symptom is uninformative.
# Nothing is submitted.
#
# v2 changes:
#   * step 0: verify that the NEW Hydra layer is actually installed. Without it, step 4
#     composes the old configs/config_hydra.yaml, whose hydra: block is merged onto
#     BasicLauncherConf, and fails with
#         Key 'submitit_folder' not in 'BasicLauncherConf'
#     which says nothing about Jean Zay.
#   * fixed a command-substitution bug: backticks inside a double-quoted echo were
#     executing sacctmgr instead of printing it.

cd "$(dirname "$0")/.."
ok=0; ko=0
pass() { echo "  PASS  $1"; ok=$((ok+1)); }
fail() { echo "  FAIL  $1"; ko=$((ko+1)); }

echo "== 0. installation of the new Hydra layer =="
echo "      repo root: $(pwd)"
if [ ! -f main_hydra.py ]; then
    fail "main_hydra.py not found here -- preflight.sh is not in <repo>/slurm/"
elif grep -q 'config_name="config_hydra"' main_hydra.py; then
    fail "main_hydra.py is the OLD one (config_name=\"config_hydra\"): it composes
        configs/config_hydra.yaml, which has no 'override hydra/launcher' and therefore
        merges Slurm fields onto BasicLauncherConf. Install the new main_hydra.py."
elif grep -q 'config_name="config"' main_hydra.py; then
    pass "main_hydra.py uses config_name=\"config\""
else
    fail "main_hydra.py: cannot identify config_name -- check it by hand"
fi
for f in configs/config.yaml configs/machine/jz_v100_t3.yaml \
         configs/machine/jz_v100_dev.yaml; do
    if [ -f "$f" ]; then pass "$f"; else fail "$f is missing"; fi
done
if grep -q 'override hydra/launcher' configs/config.yaml 2>/dev/null; then
    pass "configs/config.yaml selects the submitit_slurm launcher schema"
else
    fail "configs/config.yaml has no 'override hydra/launcher: submitit_slurm'"
fi
for f in configs/config_hydra.yaml configs/mlxp.yaml configs/mlxpy.yaml; do
    [ -f "$f" ] && echo "  WARN  $f still present; delete it to avoid composing it by accident"
done

echo "== 1. project / accounting =="
if [ -n "${GRNEWT_PROJECT:-}" ]; then pass "GRNEWT_PROJECT=$GRNEWT_PROJECT"
else fail "GRNEWT_PROJECT unset -> account: null -> every sbatch is rejected"; fi
if command -v idrproj >/dev/null 2>&1; then
    echo "  --- idrproj (check the code and the @v100/@a100/@h100 hours you own) ---"
    idrproj 2>&1 | sed 's/^/      /'
else
    echo '      (idrproj not found; try idracct, or: sacctmgr show assoc user=$USER)'
fi

echo "== 2. disk spaces =="
for v in HOME WORK SCRATCH DSDIR; do
    p="${!v:-}"
    if [ -z "$p" ]; then fail "\$$v is not defined"
    elif [ ! -d "$p" ]; then fail "\$$v=$p does not exist (post-Lustre-migration path?)"
    else
        case "$p" in
            /gpfswork*|/gpfsdswork*|/gpfsstore*|/gpfsscratch*)
                fail "\$$v=$p is an OLD Spectrum Scale path; expected /lustre/...";;
            *) pass "\$$v=$p";;
        esac
    fi
done
if [ -n "${WORK:-}" ] && touch "$WORK/.grnewt_write_test" 2>/dev/null; then
    rm -f "$WORK/.grnewt_write_test"; pass "\$WORK is writable"
else
    fail "\$WORK not writable (quota? broken symlink?) -- submitit cannot create its folder"
fi
for l in "$HOME/.local" "$HOME/.conda"; do
    if [ -L "$l" ] && [ ! -e "$l" ]; then
        fail "$l is a BROKEN symlink (typical leftover of the 2024 WORK migration)"
    fi
done

RES="${GRNEWT_RESULTS:-}"
case "$RES" in
    "${STORE:-@@none@@}"*|/lustre/fsstor/*)
        fail "GRNEWT_RESULTS=$RES is under \$STORE, unreachable from compute nodes
        since 2024-07-22: the job cannot even open its log file. Use \$SCRATCH." ;;
    /*) pass "GRNEWT_RESULTS=$RES (absolute, outside STORE)" ;;
    "") fail "GRNEWT_RESULTS is not set" ;;
    *)  fail "GRNEWT_RESULTS=$RES is not an absolute path" ;;
esac
echo "  NOTE  the write test below runs on the FRONT-END. A path writable here can"
echo "        still be unreachable from a compute node (STORE). Confirm with:"
echo "          ./slurm/interactive.sh v100 1   then   touch \$GRNEWT_RESULTS/probe"

echo "== 3. python environment =="
python - <<'PY'
import sys, importlib
print(f"      sys.executable = {sys.executable}")
print("      NB: submitit hardcodes THIS path into the sbatch script.")
for m in ("torch", "torchvision", "hydra", "omegaconf", "submitit",
          "hydra_plugins.hydra_submitit_launcher.submitit_launcher", "grnewt"):
    try:
        mod = importlib.import_module(m)
        print(f"  PASS  import {m}  ({getattr(mod, '__file__', '?')})")
    except Exception as e:
        print(f"  FAIL  import {m}: {e!r}")
PY

echo "== 4. Hydra composition (no submission) =="
# NB: `--cfg hydra --resolve` cannot be used here: Hydra strips every non-hydra
# top-level key from that view, so ${paths.results} becomes unresolvable. See
# slurm/check_config.py.
if HYDRA_FULL_ERROR=1 python slurm/check_config.py run_id=preflight \
        >/tmp/grnewt_launcher.$$ 2>&1; then
    pass "hydra.launcher composes and resolves"
    sed 's/^/      /' /tmp/grnewt_launcher.$$
else
    fail "hydra.launcher does not compose or does not resolve:"
    sed 's/^/      /' /tmp/grnewt_launcher.$$
    echo "      If the message mentions BasicLauncherConf, re-read step 0."
fi
rm -f /tmp/grnewt_launcher.$$

echo "== 5. dataset =="
if [ -n "${DSDIR:-}" ]; then
    ls "$DSDIR" 2>/dev/null | grep -i -E 'cifar|mnist|imagenet' | sed 's/^/      /' \
        || echo "      (nothing matching cifar/mnist/imagenet directly under \$DSDIR)"
    echo "      torchvision expects <root>/cifar-10-batches-py and <root>/MNIST/raw."
    echo "      download=False in datasets.py, and compute nodes have no internet:"
    echo "      a wrong root fails at epoch 0, after the allocation is granted."
fi

echo
echo "== $ok passed, $ko failed =="
[ "$ko" -eq 0 ] || echo "Fix the failures above before submitting anything."
