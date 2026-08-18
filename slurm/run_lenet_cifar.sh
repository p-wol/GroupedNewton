#!/bin/bash
# slurm/run_lenet_cifar.sh -- LeNet-5 / CIFAR-10, Adam + Nesterov lrs, lr grid.
# Only experiment parameters here; everything cluster-related is in configs/cluster/.
set -euo pipefail

exec "$(dirname "$0")/submit.sh" \
    paths=local_perso \
    cluster=none \
    expe_series='LeNet_CIFAR_02_rerun_expes' \
    seed=571677914 \
    system.dtype=32 \
    model.name='LeNet' \
    model.args='6-16-120-84-10' \
    model.act_function='tanh' \
    model.scaling=False \
    model.init.sigma_w=1. \
    model.init.sigma_b=0. \
    dataset.name='CIFAR10' \
    dataset.valid_size=5000 \
    dataset.batch_size=100 \
    dataset.data_augm=False \
    logs_hg.use=False \
    logs_hg.batch_size=1000 \
    optimizer.epochs=200 \
    optimizer.name='NewtonSummaryUniformAvg' \
    optimizer.lr=.003 \
    optimizer.weight_decay=0. \
    optimizer.momentum=.9 \
    optimizer.hg.batch_size=100 \
    optimizer.hg.partition='canonical' \
    optimizer.hg.damping=.3 \
    optimizer.hg.period_hg=1 \
    optimizer.hg.remove_negative=True \
    optimizer.hg.updater.name='SGD' \
    optimizer.hg.updater.momentum=.9 \
    optimizer.hg.updater.momentum_damp=.0 \
    optimizer.hg.nesterov.use=True \
    optimizer.hg.nesterov.damping_int=1. \
    optimizer.hg.uniform_avg.period=3 \
    optimizer.hg.uniform_avg.warmup=3 \
    optimizer.hg.dmp_auto.use=True \
    optimizer.hg.dmp_auto.patience=2 \
    optimizer.hg.dmp_auto.threshold=.0001 \
    optimizer.hg.dmp_auto.factor=.5 \
    "$@"

# Notes:
#  * dataset.path is no longer passed: it comes from configs/paths/jz.yaml ($DSDIR).
#  * hydra.launcher.* is no longer passed: cluster=<name> selects a validated preset.
#  * to smoke-test this exact grid without training:
#        ./slurm/run_lenet_cifar.sh cluster=jz_v100_dev dry_run=true optimizer.epochs=1
#  * "$@" at the end lets you override anything ad hoc from the command line.
