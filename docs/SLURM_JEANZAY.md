# Hydra + submitit + Slurm on Jean Zay — rewritten layer

Scope: the files below replace the Hydra/Slurm layer only. `training_hydra.py`,
`src/grnewt/` and the tests are untouched.

```
configs/config.yaml              # primary config; NO launcher settings, NO ${now:}
configs/paths/{jz,local}.yaml    # $WORK / $SCRATCH / $DSDIR, no hard-coded absolute path
configs/cluster/jz_v100_t3.yaml  # \
configs/cluster/jz_v100_dev.yaml #  } package hydra.launcher, merged on the typed
configs/cluster/jz_a100.yaml     #  } SlurmQueueConf schema -> typos are errors
configs/cluster/jz_h100.yaml     # /
main_hydra.py                    # env probe, dry_run, failure captured in the job dir
slurm/preflight.sh               # login-node checks, submits nothing
slurm/submit.sh                  # the only submission entry point
slurm/run_lenet_cifar.sh         # example experiment script
```

Deleted / superseded: `configs/config_hydra.yaml`, `configs/mlxp.yaml`,
`configs/mlxpy.yaml`, `run_*_hydra.sh`.

---

## 1. What was wrong in the previous layer

Facts about the files in the uploaded zip (no commit hash was supplied).

| # | Fact | Consequence |
|---|---|---|
| A1 | `hydra.launcher.account: null` | On Jean Zay the account is mandatory: IDRIS documents `--account=<proj>@v100` for V100 hours, `@a100`, `@h100`, `@cpu` otherwise. With no account and hours on several accounting types, `sbatch` rejects the submission. **First suspect.** |
| A2 | `parent_dir='/gpfswork/rech/tza/uki35ex/...'` in all five `run_*_hydra.sh`, and `submitit_folder` under it | `/gpfswork` is the pre-August-2024 Spectrum Scale path. IDRIS kept compatibility symlinks "at least to begin with" and asks that they be replaced by `$WORK` (now `/lustre/fswork/projects/rech/...`). If the symlink is gone, submitit cannot create its folder and Slurm cannot open the log file: the job fails with **no readable output at all**, which matches "the error code is very vague". **Second suspect.** |
| A3 | `gpus_per_node: 1` **and** `gres: 'gpu:1'` are both set | submitit historically translated `gpus_per_node` to `--gres=gpu:N` but now emits `--gpus-per-node`. Two GPU directives are then written in the same script. `--gpus-per-node` is only supported by Slurm's `select/cons_tres` plugin, so on a site that does not enable it the job is rejected. IDRIS documents `--gres=gpu:N`. |
| A4 | `partition: 'gpu_p13'` hard-coded | The current IDRIS English page no longer names this partition: the 4×V100 nodes are the *default* GPU partition ("default (no option)" in the summary table). The name still appears in the French pages, so it probably still resolves — but naming it buys nothing and is a needless dependency. |
| A5 | `qos: 'qos_gpu-dev'` with `array_parallelism: 256` | `qos_gpu-dev` allows **at most 10 jobs running or pending per user**, 2 h. Slurm counts array tasks individually, so any sweep with ≥ 11 configurations is rejected at submission under this QoS. Not triggered by the current 3-point grids; will be triggered by O5 (grid × 3 seeds). |
| A6 | No `--hint=nomultithread`, no `setup:` | The IDRIS templates set `--hint=nomultithread`. With no `setup:`, the job's module environment is whatever Slurm propagated from the login shell — implicit, and invisible in the repository. |
| A7 | `${now:%Y-%m-%d_%H-%M-%S}` inside `hydra.run.dir`, `hydra.sweep.dir` and hence `submitit_folder` | The launcher resolves this on the login node; the job re-composes the config in a fresh process on the compute node. The `now` resolver is cached *per process*, so the job's `hydra.sweep.dir` can carry a different timestamp than the directory the launcher created. Cheap to eliminate, so eliminated: `run_id` is a CLI override. |
| A8 | `configs/config_hydra.yaml` has no `defaults:` list; the whole `SlurmQueueConf` is copied into the primary config's `hydra:` block | The config is only valid when the command line also says `hydra/launcher=submitit_slurm`. `python main_hydra.py` alone (no `--multirun`) is not a supported path. Also, since the block is merged untyped, a misspelled field is silently ignored. |
| A9 | `assign_device` (`training_hydra.py:59`) builds `ValueError(...)` without `raise` | An out-of-range `system.device` falls through and returns an `int`, which then fails later with an unrelated message. Unrelated to Slurm, but it is in the path. |

Not a problem, checked: `datasets.py` already passes `download=False` everywhere (compute
nodes have no internet), and the trainer writes exclusively under
`HydraConfig.get().runtime.output_dir`, never to the cwd.

## 2. Ranked hypotheses for the current failure

I cannot reproduce anything from here, so these are ordered by (probability × how well
they explain "vague error"), not verified:

1. **A1** — missing account. Symptom: nothing in `.submitit/`, an `sbatch` error string
   swallowed by submitit's exception, or an immediately failed job.
2. **A2** — dead `/gpfswork` path. Symptom: job appears in `sacct` as FAILED with no log.
3. **Interpreter path.** submitit hardcodes the submitting process's `sys.executable`
   into the sbatch script. If you submit from a conda env whose prefix lives under
   `$HOME/.local` or `$HOME/.conda` symlinked to the old `$WORK`, the compute node runs a
   path that does not resolve. Symptom: exit code 127/2, empty log.
4. **A3** — conflicting GPU directives.
5. **Plugin not visible in the job.** `hydra-submitit-launcher` installed with
   `pip install --user` under a broken `$HOME/.local` link.

Each is falsified or confirmed in under a minute by §4.

## 3. Validation procedure (≈ 10 minutes, ~0 GPU-hours)

```bash
export GRNEWT_PROJECT=xxx            # 3-letter code from `idrproj`, no @v100
./slurm/preflight.sh                 # submits nothing
./slurm/submit.sh cluster=jz_v100_dev expe_series=smoke dry_run=true
```

The third command submits one 1-minute job that imports torch/grnewt, prints the GPU it
was given, writes `env.json` and `DRY_RUN_OK`, and exits. If it succeeds, the entire
chain works: composition → sbatch → module environment → interpreter → CUDA → output
directory. Then, and only then:

```bash
./slurm/submit.sh cluster=jz_v100_dev expe_series=smoke optimizer.epochs=1   # real, 1 epoch
./slurm/run_lenet_cifar.sh                                                   # production
```

## 4. What to read when a job fails

In `<results>/<expe_series>/<run_id>/.submitit/<job_id>/`:

| File | What it answers |
|---|---|
| `*_submission.sh` | **Read this first.** It is the exact sbatch script. Check: the `#SBATCH --account` line exists; there is exactly one GPU directive; the `srun ... <python>` path exists on a compute node (`ls -l` it); the `module load` lines are present. |
| `*_log.out` | Job stdout (`stderr_to_stdout: true` merges both). Empty ⇒ the job never reached Python ⇒ the cause is in the sbatch script or in the log path. |
| `*_submitted.pkl` | The pickled call. Its existence proves the launcher ran. |
| `<job_dir>/env.json` | Written by `main_hydra.py` before anything heavy: host, interpreter, `SLURM_*`, `nvidia-smi`, `torch.cuda`. |
| `<job_dir>/FAILED.txt` | Python traceback, saved next to the run even if the Slurm log is lost. |

And on the login node: `sacct -j <id> --format=JobID,State,ExitCode,DerivedExitCode,Reason,Elapsed`.
`Reason` is where Slurm puts QoS/account rejections.

## 5. Deliberate design decisions

- **No local/basic launcher config.** Hydra only instantiates a launcher in `--multirun`.
  A single run (`python main_hydra.py run_id=x ...`) on an interactive GPU allocation
  ignores `hydra.launcher` entirely, so there is nothing to configure for it.
- **`cluster` is a separate group, not `hydra/launcher`.** Defining
  `configs/hydra/launcher/jz.yaml` would *replace* the launcher node and lose the schema;
  merging `configs/cluster/*.yaml` into `# @package hydra.launcher` on top of
  `override hydra/launcher: submitit_slurm` keeps the typed node, so `qos_gpu-dvv` or
  `cpu_per_task` fail at composition, on the login node.
- **`run_id: ???`.** A missing run id is a hard error, not a silently shared directory.
- **`max_num_timeout: 0`.** Requeue-on-timeout without checkpoint/resume in the trainer
  restarts from scratch and bills the hours twice. Raise it only after `Trainer` can
  resume from `last_ckpt` (the commented-out block at the top of the old `main_hydra.py`).

## 6. Claims in this document that I have *not* verified

- submitit's exact translation of `gpus_per_node`/`gres`/`additional_parameters` into
  `#SBATCH` lines (A3): RECALLED from the source, and the submitit docs confirm the
  `gpus_per_node` → `--gpus-per-node` change. Settled definitively by reading
  `*_submission.sh` once.
- The per-process caching of the `now` resolver (A7): RECALLED. Irrelevant now that
  `${now:}` is gone.
- Whether `gpu_p13` still resolves as a partition name (A4).
- The layout of `$DSDIR` for CIFAR-10/MNIST, i.e. whether `dataset.path=$DSDIR` is the
  right root for torchvision (`<root>/cifar-10-batches-py`, `<root>/MNIST/raw`). The old
  scripts used `/lustre/fsmisc/dataset`; `preflight.sh` prints the candidates.
- That `pytorch-gpu/py3/2.4.0` is still available, and that `hydra-core`,
  `hydra-submitit-launcher` and `grnewt` are importable from it. `preflight.sh` checks it.

Sources for the Jean Zay facts: IDRIS, *Jean Zay: GPU Slurm partitions* (partitions, QoS
names and limits, accounting), and *Changes and impacts related to Jean Zay H100
extension* (Lustre migration and new `$WORK`/`$SCRATCH` paths, A100 QoS renaming,
`arch/a100` and `arch/h100` modules, `--hint=nomultithread` template).
