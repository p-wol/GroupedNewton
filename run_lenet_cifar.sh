parent_work_dir="./data/.workdir"
parent_log_dir="./data/outputs/LeNet_CIFAR_01_debug"


HYDRA_FULL_ERROR=1   OC_CAUSE=1 python main.py\
                seed=1\
				system.dtype=32\
        		model.name='LeNet'\
	        	model.args='6-16-120-84-10'\
				model.act_function='tanh'\
				model.scaling=False\
				model.init.sigma_w=1.\
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
				optimizer.lr=.0001\
				optimizer.weight_decay=0.\
				optimizer.momentum=.9\
				optimizer.hg.batch_size=1000\
				optimizer.hg.optimizer='SGD'\
				optimizer.hg.partition='canonical'\
				optimizer.hg.damping=.3\
				optimizer.hg.damping_schedule='None'\
				optimizer.hg.momentum=.9\
				optimizer.hg.momentum_damp=.9\
				optimizer.hg.mom_lrs=.5\
				optimizer.hg.period_hg=10\
				optimizer.hg.nesterov.use=True\
				optimizer.hg.nesterov.damping_int=1.\
				optimizer.hg.remove_negative=True\
				optimizer.hg.dmp_auto.use=True\
				optimizer.hg.dmp_auto.patience=0\
				optimizer.hg.dmp_auto.threshold=.0001\
				optimizer.hg.dmp_auto.factor=.5\
				optimizer.kfac.stat_decay=.95\
				optimizer.kfac.damping=.03\
				optimizer.kfac.kl_clip=.01\
				optimizer.kfac.weight_decay=.003\
				optimizer.kfac.tcov=10\
				optimizer.kfac.tinv=100\
                +mlxpy.interactive_mode=True\
                +mlxpy.version_manager.parent_work_dir=$parent_work_dir\
                +mlxpy.logger.parent_log_dir=$parent_log_dir\
                +mlxpy.use_scheduler=True\
                +mlxpy.use_version_manager=False\
                +mlxpy.use_logger=True\
