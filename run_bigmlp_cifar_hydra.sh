#!/bin/bash

parent_dir='/gpfswork/rech/tza/uki35ex/_Experiments/GroupedNewton_Results'
expe_series='BigMLP_MNIST_UnifAvg_05_many_tests'

module purge
module load pytorch-gpu/py3/2.4.0


HYDRA_FULL_ERROR=1 OC_CAUSE=1 python main_hydra.py --multirun hydra/launcher=submitit_slurm\
				hydra.launcher.timeout_min=400\
        		hydra.launcher.partition='gpu_p13'\
        		hydra.launcher.qos='qos_gpu-t3'\
				parent_dir="${parent_dir}"\
				expe_series="${expe_series}"\
                seed=571677914\
				system.dtype=32\
				model.name='Perceptron'\
				model.args='20*1024'\
				model.act_function='elu'\
				model.scaling=False\
				model.init.sigma_w=1.4142\
				model.init.sigma_b=0.\
				dataset.name='CIFAR10'\
				dataset.path='/lustre/fsmisc/dataset'\
				dataset.valid_size=5000\
				dataset.batch_size=100\
				dataset.data_augm=False\
				logs_hg.use=False\
				logs_hg.batch_size=1000\
				logs_hg.test_float=False\
				optimizer.epochs=100\
				optimizer.name='NewtonSummaryUniformAvg'\
				optimizer.lr=.001\
				optimizer.weight_decay=0.\
				optimizer.momentum=.9\
				optimizer.hg.batch_size=100\
				optimizer.hg.partition='canonical'\
				optimizer.hg.damping=.3\
				optimizer.hg.period_hg=5\
				optimizer.hg.remove_negative=True\
				optimizer.hg.updater.name='SGD'\
				optimizer.hg.updater.momentum=.9\
				optimizer.hg.updater.momentum_damp=.0\
				optimizer.hg.nesterov.use=True\
				optimizer.hg.nesterov.damping_int=10.\
        		optimizer.hg.uniform_avg.period=5\
        		optimizer.hg.uniform_avg.warmup=5\
				optimizer.hg.dmp_auto.use=True\
				optimizer.hg.dmp_auto.patience=2\
				optimizer.hg.dmp_auto.threshold=.0001\
				optimizer.hg.dmp_auto.factor=.5\
				optimizer.kfac.stat_decay=.95\
				optimizer.kfac.damping=.03\
				optimizer.kfac.kl_clip=.01\
				optimizer.kfac.weight_decay=.003\
				optimizer.kfac.tcov=10\
				optimizer.kfac.tinv=100\

