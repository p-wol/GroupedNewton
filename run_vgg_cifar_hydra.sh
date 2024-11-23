#!/bin/bash

parent_dir='/gpfswork/rech/tza/uki35ex/_Experiments/GroupedNewton_Results'
expe_series='VGG_CIFAR_ICLR25_01_test'

module purge
module load pytorch-gpu/py3/2.3.0


HYDRA_FULL_ERROR=1 OC_CAUSE=1 python main_hydra.py --multirun hydra/launcher=submitit_slurm\
				hydra.launcher.timeout_min=240\
        		hydra.launcher.partition='gpu_p13'\
        		hydra.launcher.qos='qos_gpu-t3'\
				parent_dir="${parent_dir}"\
				expe_series="${expe_series}"\
                seed=571677914\
				system.dtype=32\
	        	model.name='VGG'\
		        model.args='A'\
				model.act_function='elu'\
				model.scaling=False\
				model.init.sigma_w=1.4142\
				model.init.sigma_b=0.\
				dataset.name='CIFAR10'\
				dataset.path='/gpfswork/rech/tza/uki35ex/dataset'\
				dataset.valid_size=5000\
				dataset.batch_size=100\
				logs_hg.use=False\
				logs_hg.batch_size=1000\
				logs_hg.test_float=False\
				optimizer.epochs=100\
				optimizer.name='NewtonSummary'\
				optimizer.lr=.003\
				optimizer.weight_decay=0.\
				optimizer.momentum=.9\
				optimizer.hg.batch_size=200\
				optimizer.hg.optimizer='SGD'\
				optimizer.hg.partition='canonical'\
				optimizer.hg.damping=.3\
				optimizer.hg.damping_schedule='None'\
				optimizer.hg.momentum=.9\
				optimizer.hg.momentum_damp=.9\
				optimizer.hg.mom_lrs=0.\
				optimizer.hg.period_hg=10\
				optimizer.hg.nesterov.use=True\
				optimizer.hg.nesterov.damping_int=3.\
				optimizer.hg.remove_negative=True\
        		optimizer.hg.uniform_avg.use=False\
        		optimizer.hg.uniform_avg.period=5\
        		optimizer.hg.uniform_avg.warmup=5\
				optimizer.hg.dmp_auto.use=True\
				optimizer.hg.dmp_auto.patience=0\
				optimizer.hg.dmp_auto.threshold=.0001\
				optimizer.hg.dmp_auto.factor=.8\
				optimizer.kfac.stat_decay=.95\
				optimizer.kfac.damping=.03\
				optimizer.kfac.kl_clip=.01\
				optimizer.kfac.weight_decay=.003\
				optimizer.kfac.tcov=10\
				optimizer.kfac.tinv=100\
