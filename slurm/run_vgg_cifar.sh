#!/bin/bash
# slurm/run_vgg_cifar.sh -- VGG, Adam + Nesterov lrs, lr grid.
# Only experiment parameters here; everything machine-related is in configs/machine/.
#
# The overrides live in an ARRAY, not in a backslash-continued command line.  A `#`
# comment inside a `\`-continued command silently truncates it: the backslash at the
# end of the commented line is part of the comment, so the command ends there and
# every later argument -- including "$@" -- is dropped.  Inside `( ... )` a newline is
# just a separator, so individual overrides can be commented out safely.
set -euo pipefail

args=(
    machine=laptop
    expe_series='VGG_CIFAR_00_basic_tests'
    seed=571677914
    system.dtype=32
    model.name='VGG'
    model.args='A'
    model.act_function='elu'
    model.scaling=False
    model.init.sigma_w=1.4142
    model.init.sigma_b=0.
    dataset.name='CIFAR10'
    dataset.valid_size=5000
    dataset.batch_size=100
    dataset.data_augm=False
    logs_hg.use=False
    logs_hg.batch_size=1000
    optimizer.epochs=20
    optimizer.name='NewtonSummaryUniformAvg'
    optimizer.lr=.003
    optimizer.weight_decay=0.
    optimizer.momentum=.9
    optimizer.hg.batch_size=100
    optimizer.hg.partition='canonical'
    optimizer.hg.damping=.3
    optimizer.hg.period_hg=10
    optimizer.hg.normalize_dirs=True
    optimizer.hg.remove_negative=True
    optimizer.hg.updater.name='SGD'
    optimizer.hg.updater.momentum=.9
    optimizer.hg.updater.momentum_damp=.0
    optimizer.hg.nesterov.use=True
    optimizer.hg.nesterov.damping_int=1.
    optimizer.hg.uniform_avg.period=3
    optimizer.hg.uniform_avg.warmup=3
    #optimizer.hg.static_avg.nsamples=5
    optimizer.hg.dmp_auto.use=True
    optimizer.hg.dmp_auto.patience=2
    optimizer.hg.dmp_auto.threshold=.0001
    optimizer.hg.dmp_auto.factor=.5
)

exec "$(dirname "$0")/submit.sh" "${args[@]}" "$@"
