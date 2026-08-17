#!/bin/bash
# slurm/preflight.sh -- run this on a Jean Zay login node before anything else.
# Each check corresponds to a failure mode whose Slurm-side symptom is uninformative.
# Nothing is submitted.

cd "$(dirname "$0")/.."
ok=0; ko=0
pass() { echo "  PASS  $1"; ok=$((ok+1)); }
fail() { echo "  FAIL  $1"; ko=$((ko+1)); }

echo "== 1. project / accounting =="
if [ -n "${GRNEWT_PROJECT:-}" ]; then pass "GRNEWT_PROJECT=$GRNEWT_PROJECT"
else fail "GRNEWT_PROJECT unset -> account: null -> every sbatch is rejected"; fi
if command -v idrproj >/dev/null 2>&1; then
    echo "  --- idrproj (check the code and the @v100/@a100/@h100 hours you own) ---"
    idrproj 2>&1 | sed 's/^/      /'
else
    echo "  (idrproj not found; use idracct or `sacctmgr show assoc user=$USER`)"
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
if HYDRA_FULL_ERROR=1 python main_hydra.py --cfg hydra --package hydra.launcher \
        --resolve run_id=preflight >/tmp/grnewt_launcher.$$ 2>&1; then
    pass "hydra.launcher composes and resolves"
    sed 's/^/      /' /tmp/grnewt_launcher.$$
else
    fail "hydra.launcher does not compose:"; sed 's/^/      /' /tmp/grnewt_launcher.$$
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
