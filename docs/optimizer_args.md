# `optimizer.hg` reference

Generated from `src/grnewt/config.py` by `python -m grnewt.config`.
Do not edit by hand: `tests/test_config_schema.py` fails if it drifts.

| field | type | default | read by | description |
|---|---|---|---|---|
| `batch_size` | `int` | `-1` | all | batch size for the (H, g) estimation; -1 = dataset batch size |
| `partition` | `Partition` | `'canonical'` | all | group construction rule |
| `partition_arg` | `int | None` | `None` | all | integer argument of partition in {blocks, alternate}; unused otherwise |
| `partition_str` | `str | None` | `None` | all | string argument of partition in {vgg, perceptron} |
| `diagonal` | `bool` | `False` | NewtonSummary, NewtonSummaryUniformAvg | compute only the diagonal of Hbar |
| `noregul` | `bool` | `False` | NewtonSummary, NewtonSummaryUniformAvg | bypass every regularization: lrs = Hbar^{-1} gbar |
| `ridge` | `float` | `0.0` | all | ridge added to Hbar when nesterov.use is False |
| `damping` | `float` | `1.0` | all | per-group damping; multiplies the computed lr |
| `period_hg` | `int` | `1` | NewtonSummary, NewtonSummaryUniformAvg | training steps between two recomputations of (H, g) |
| `mom_lrs` | `float` | `0.0` | NewtonSummary, NewtonSummaryUniformAvg | momentum on the learning rates |
| `movavg` | `float` | `0.0` | NewtonSummary | moving average on (H, g) |
| `maintain_true_lrs` | `bool` | `True` | NewtonSummary | keep the unclipped lrs as the momentum state |
| `remove_negative` | `bool` | `False` | NewtonSummary, NewtonSummaryUniformAvg | clamp negative learning rates to zero |
| `hg_batched` | `bool` | `False` | NewtonSummaryUniformAvg | use the batched version of compute_Hg |
| `hg_batched_chunk` | `int` | `-1` | NewtonSummaryUniformAvg | chunk_size in the batched version of compute_hg; -1 = S (partition size) |
| `nologs` | `bool` | `False` | all | do not dump the (H, g, lrs) logs |
| `damping_schedule.use` | `bool` | `False` | all | geometrically decay `damping` over the first `epoch` epochs |
| `damping_schedule.final` | `float` | `1.0` | all | target value of `damping` at epoch `epoch` |
| `damping_schedule.epoch` | `int` | `0` | all | epoch at which `damping` reaches `final` |
| `updater.name` | `UpdaterName` | `'SGD'` | all | inner optimizer producing the search direction u |
| `updater.momentum` | `float` | `0.9` | all | momentum of the updater (SGD only) |
| `updater.momentum_damp` | `float` | `0.0` | all | dampening of the updater (SGD only) |
| `nesterov.use` | `bool` | `False` | all | solve the anisotropic cubic subproblem instead of H^{-1} g |
| `nesterov.damping_int` | `float` | `1.0` | all | lambda_int >= 0; strength of the cubic regularization |
| `nesterov.mom_order3_` | `float` | `0.0` | NewtonSummary | EMA coefficient on order3_ = |order3|^(1/3) |
| `nesterov.threshold_D_sing` | `float` | `0.0` | all | relative threshold below which d_i counts as zero; 0.0 is the only affine-invariant choice (nesterov.py, Appendix E) |
| `nesterov.hard_case_rtol` | `float` | `1e-12` | all | relative tolerance defining the near-null eigenspace of K + c x0 I in the hard case |
| `nesterov.refine` | `bool` | `False` | all | safeguarded Newton refinement of (r, eta); NOT validated over the fuzz set (STATE.md, N3) |
| `uniform_avg.period` | `int` | `1` | NewtonSummaryUniformAvg | Hg-update steps between two swaps of (X^a, X^b) |
| `uniform_avg.warmup` | `int` | `0` | NewtonSummaryUniformAvg | Hg-update steps during which H, g, D are averaged but the network is not trained |
| `dmp_auto.use` | `bool` | `False` | all | reduce damping on plateau |
| `dmp_auto.apply_to` | `str` | `'damping'` | all | attribute the scheduler acts on |
| `dmp_auto.patience` | `int` | `1` | all | scheduler patience, in epochs |
| `dmp_auto.cooldown` | `int` | `0` | all | scheduler cooldown, in epochs |
| `dmp_auto.threshold` | `float` | `0.9` | all | relative improvement below which a step counts as a plateau |
| `dmp_auto.factor` | `float` | `0.9` | all | multiplicative factor applied on plateau |
