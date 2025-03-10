import random
import time
import copy
import os
from collections import OrderedDict
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.utils import data
#from kfac.optimizers import KFACOptimizer
from grnewt import compute_Hg, compute_Hg_fullbatch, fullbatch_gradient, NewtonSummary, NewtonSummaryFB, NewtonSummaryUniformAvg
from grnewt import ParamStructure, diff_n, diff_n_fullbatch
from grnewt import partition as build_partition
from grnewt.models import Perceptron, LeNet, VGG, AutoencoderMLP, Rosenbrock, RosenbrockT
from grnewt.datasets import build_MNIST, build_CIFAR10, build_ImageNet, build_toy_regression, build_None
from grnewt.nesterov import nesterov_lrs
from grnewt import ReduceDampingOnPlateau
from grnewt import optimizers, loader_pre_hooks

def set_seeds(seed):    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def assign_device(device):
    device = int(device)
    if device > -1:
        if torch.cuda.is_available():
            device = 'cuda:' + str(device)
        else:
            device = 'cpu'
    elif device == -1:
        device = 'cuda'
    elif device == -2:
        device = 'cpu'
    else:
        ValueError('Unknown device: {}'.format(device))

    return device

def get_dtype(dtype):
    if dtype == 64:
        return torch.double
    elif dtype == 32:
        return torch.float
    else:
        raise ValueError('Unknown dtype: {}'.format(dtype))


class Trainer:
    def __init__(self, config, hydra_path):
        self.args = config

        set_seeds(self.args.seed)

        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)

        self.path_metrics = f"{hydra_path}/metrics"
        if not os.path.isdir(self.path_metrics):
            os.makedirs(self.path_metrics)
        open(f"{self.path_metrics}/metrics.json", 'w').close()

        self.path_artifacts = f"{hydra_path}/artifacts"
        if not os.path.isdir(self.path_artifacts):
            os.makedirs(self.path_artifacts)

        print(self.args)

    def build_datasets(self):
        """
        Sets the following attributes:
            'train_size'
            'valid_size'
            'test_size'
            'tvsize': train_size + valid_size

            'trainset'
            'validset'
            'testset'
            'tvset': union of 'trainset' and 'validset'

            'train_loader'
            'valid_loader'
            'test_loader'

            'classification': True if classification task
            'n_classes'
            'n_channels'
            'image_size': height or width of an input image
            'channel_size': image_size**2
            'input_size': size of an input sample
            'topk_acc'
            'loss_fn'
        """

        args = self.args
        dct = {'dtype': self.dtype, 'device': self.device}

        if args.dataset.name == 'MNIST':
            self.loader_pre_hook = lambda x, y: (x.to(device = self.device, dtype = self.dtype), y.to(self.device))
            dct = build_MNIST(args, dct)
        elif args.dataset.name == 'CIFAR10':
            self.loader_pre_hook = lambda x, y: (x.to(device = self.device, dtype = self.dtype), y.to(self.device))
            dct = build_CIFAR10(args, dct)
        elif args.dataset.name == 'ImageNet':
            self.loader_pre_hook = lambda x, y: (x.to(device = self.device, dtype = self.dtype), y.to(self.device))
            dct = build_ImageNet(args, dct)
        elif args.dataset.name == 'ToyRegression':
            dct = build_toy_regression(args, dct)
        elif args.dataset.name == 'None':
            dct = build_None(args, dct)
        else:
            raise NotImplementedError('Unknown dataset: {}.'.format(args.dataset.name))

        dct.pop('dtype')
        dct.pop('device')

        for k, v in dct.items():
            setattr(self, k, v)

    def build_model(self):
        args = self.args

        # Activation function
        dct_act_functions = {'identity': lambda x: x, 
                             'tanh': torch.tanh, 
                             'relu': torch.relu, 
                             'sigmoid': torch.sigmoid, 
                             'elu': torch.nn.functional.elu}
        act_function = dct_act_functions[args.model.act_function]

        # Create model
        model_args = args.model.args
        if '*' in model_args:
            n_layers = int(args.model.args[:args.model.args.find('*')])
            n_neurons = int(args.model.args[args.model.args.find('*') + 1:])
            model_args = '-'.join([str(n_neurons) for i in range(n_layers)]) + '-{}'.format(self.n_classes)

        sigma_w = args.model.init.sigma_w
        sigma_b = args.model.init.sigma_b
        if args.model.name == 'Perceptron':
            layers = [int(s) for s in model_args.split('-')]
            layers = [self.input_size] + layers
            classification = True
            if args.dataset.name == 'ToyRegression':
                classification = False

            model = Perceptron(layers, act_function, scaling = args.model.scaling, sigma_w = sigma_w, sigma_b = sigma_b,
                    classification = classification)
        elif args.model.name == 'LeNet':
            layers = [int(s) for s in model_args.split('-')]
            layers = [self.n_channels] + layers

            model = LeNet(layers, act_function, scaling = args.model.scaling, sigma_w = sigma_w, sigma_b = sigma_b)
        elif args.model.name == 'VGG':
            vgg_setup = [s for s in model_args.split('-')]
            vgg_type = vgg_setup[0][0]
            with_batch_norm = True if 'bn' in vgg_setup[0] else False
            fc_sizes = [int(s) for s in vgg_setup[1:]]

            model = VGG(vgg_type, fc_sizes = fc_sizes, image_size = self.image_size, num_classes = self.n_classes,
                        scaling = args.model.scaling, sigma_w = sigma_w, name_act_function = args.model.act_function,
                        batch_norm = with_batch_norm)
        elif args.model.name == "ResNet":
            model_name = "resnet" + model_args
            model = torchvision.models.__dict__[model_name]
        elif args.model.name == 'AutoencoderMLP':
            layers = [int(s) for s in model_args.split('-')]
            layers = [self.input_size] + layers
            classification = False

            model = AutoencoderMLP(layers, act_function, scaling = args.model.scaling, sigma_w = sigma_w, sigma_b = sigma_b)
        elif args.model.name == 'Rosenbrock':
            lst_params = model_args.split('-')
            d = int(lst_params[0])
            a = float(lst_params[1])
            b = float(lst_params[2])

            model = Rosenbrock(d, a, b)
        elif args.model.name == 'RosenbrockT':
            lst_params = model_args.split('-')
            d = int(lst_params[0])
            a = float(lst_params[1])
            b = float(lst_params[2])

            model = RosenbrockT(d, a, b)
        else:
            raise NotImplementedError('Unknown model: {}.'.format(args.model.name))

        return model.to(device = self.device, dtype = self.dtype)

    def build_optimizer(self, model):
        args = self.args
        args_hg = self.args.optimizer.hg

        # Define useful variables
        def full_loss(x, y): 
            return self.loss_fn(self.model(x), y)

        # Build data loader for Hg
        if args_hg.batch_size == -1:
            hg_batch_size = args.dataset.batch_size
        else:
            hg_batch_size = args_hg.batch_size
        self.hg_loader = data.DataLoader(self.trainset, hg_batch_size)

        # Build partition
        if args_hg.partition == 'canonical':
            param_groups, name_groups = build_partition.canonical(model)
        elif args_hg.partition == 'wb':
            param_groups, name_groups = build_partition.wb(model)
        elif args_hg.partition == 'trivial':
            param_groups, name_groups = build_partition.trivial(model)
        elif args_hg.partition.find('blocks') == 0:
            param_groups, name_groups = build_partition.blocks(model, int(args_hg.partition[len('blocks-'):]))
        elif args_hg.partition.find('alternate') == 0:
            alternate = int(args_hg.partition[len('alternate-'):])
            if args.model.name == 'Perceptron':
                nlayers = len(model.layers)
            elif args.model.name == 'VGG':
                nlayers = len(model.features)
            lst_names_w = [['{}.weight'.format(i) for i in range(nlayers) if i % alternate == r] for r in range(alternate)]
            lst_names_b = [['{}.bias'.format(i) for i in range(nlayers) if i % alternate == r] for r in range(alternate)]
            param_groups, name_groups = build_partition.names_by_lst(model, lst_names_w + lst_names_b)
        elif args_hg.partition.find('vgg') == 0:
            partition_args = args_hg.partition[len('vgg-'):]
            param_groups, name_groups = model.partition(partition_args)
        elif args_hg.partition.find('perceptron') == 0:
            partition_args = args_hg.partition[len('perceptron-'):]
            param_groups, name_groups = model.partition(partition_args)
        else:
            raise NotImplementedError('Unknown partition.')
        #param_groups = ParamStructure(pgroups)

        print(name_groups)

        # Build parameters for Nesterov
        dct_nesterov = None
        if args_hg.nesterov.use:
            dct_nesterov = {'use': args_hg.nesterov.use,
                            'damping_int': args_hg.nesterov.damping_int,
                            'mom_order3_': args_hg.nesterov.mom_order3_}
        self.dct_nesterov = dct_nesterov

        # Build parameters for uniform average
        dct_uniform_avg = None
        if args.optimizer.name == "NewtonSummaryUniformAvg":
            dct_uniform_avg = {'period': args_hg.uniform_avg.period,
                               'warmup': args_hg.uniform_avg.warmup}
        self.dct_uniform_avg = dct_uniform_avg

        # Build optimizer
        if args.optimizer.name == 'SGD':
            optimizer = optim.SGD(param_groups, lr = args.optimizer.lr, 
                    momentum = args.optimizer.momentum, weight_decay = args.optimizer.weight_decay)
        elif args.optimizer.name == 'Adam':
            optimizer = optim.Adam(param_groups, lr = args.optimizer.lr)
        elif args.optimizer.name.find("NewtonSummary") == 0:
            if args_hg.updater.name == "SGD":
                updater = optimizers.SGDUpdate(model.parameters(), lr = 1, momentum = args_hg.updater.momentum, dampening = args_hg.updater.momentum_damp)
            elif args_hg.updater.name == "Adam":
                updater = optimizers.AdamUpdate(model.parameters(), lr = 1)
            else:
                raise NotImplementedError(f"Unknown updater: {args_hg.updater.name}, expected 'SGD' or 'Adam'.")

            if args.optimizer.name == 'NewtonSummary':
                optimizer = NewtonSummary(param_groups, full_loss, self.hg_loader, updater,
                        loader_pre_hook = self.loader_pre_hook,
                        damping = args_hg.damping, period_hg = args_hg.period_hg, mom_lrs = args_hg.mom_lrs, 
                        dct_nesterov = dct_nesterov, 
                        movavg = args_hg.movavg, ridge = args_hg.ridge, 
                        remove_negative = args_hg.remove_negative,
                        maintain_true_lrs = args_hg.maintain_true_lrs, diagonal = args_hg.diagonal)
            elif args.optimizer.name == 'NewtonSummaryFB':
                optimizer = NewtonSummaryFB(param_groups, full_loss, self.model, self.loss_fn,
                        self.hg_loader, self.train_size,
                        damping = args_hg.damping, ridge = args_hg.ridge, 
                        dct_nesterov = dct_nesterov, loader_pre_hook = self.loader_pre_hook)
            elif args.optimizer.name == "NewtonSummaryUniformAvg":
                optimizer = NewtonSummaryUniformAvg(param_groups, full_loss, self.hg_loader, updater, 
                        loader_pre_hook = self.loader_pre_hook,
                        damping = args_hg.damping, period_hg = args_hg.period_hg, mom_lrs = args_hg.mom_lrs,
                        dct_nesterov = dct_nesterov, 
                        remove_negative = args_hg.remove_negative, dct_uniform_avg = dct_uniform_avg)
        elif args.optimizer.name == 'KFAC':
            optimizer = KFACOptimizer(self.model,
                    lr = args.optimizer.lr,
                    momentum = args.optimizer.momentum,
                    stat_decay = args.optimizer.kfac.stat_decay,
                    damping = args.optimizer.kfac.damping,
                    kl_clip = args.optimizer.kfac.kl_clip,
                    weight_decay = args.optimizer.kfac.weight_decay,
                    TCov = args.optimizer.kfac.tcov,
                    TInv = args.optimizer.kfac.tinv)
        elif args.optimizer.name == 'LBFGS':
            if args.optimizer.lbfgs.line_search_fn == 'none':
                line_search_fn = None
            else:
                line_search_fn = args.optimizer.lbfgs.line_search_fn

            optimizer = optim.LBFGS(param_groups, lr = args.optimizer.lr, 
                    max_iter = args.optimizer.lbfgs.max_iter,
                    history_size = args.optimizer.lbfgs.history_size,
                    line_search_fn = line_search_fn)
        else:
            raise NotImplementedError('Unknown optimizer: {}.'.format(args.optimizer.name))

        # Store grouping data
        self.name_groups = name_groups
        self.param_groups = param_groups
        self.param_struct = ParamStructure(self.param_groups)
        self.tup_params = self.param_struct.tup_params

        self.full_loss = full_loss

        return optimizer

    def test_model(self, loader, dsname):
        with torch.no_grad():
            cum_nll = 0
            cum_pen = 0
            cum_loss = 0
            total = 0
            if self.classification:
                correct = [torch.tensor([0], dtype = self.dtype, device = self.device) for i in range(len(self.topk_acc))]
            for i, (images, labels) in enumerate(loader):
                # Convert torch tensor to Variable
                images, labels = self.loader_pre_hook(images, labels)

                # Forward
                outputs = self.model(images)
                nll = self.loss_fn(outputs, labels)
                pen = torch.tensor(0.)
                loss = nll + pen

                cum_nll += nll.item()
                cum_pen += pen.item()
                cum_loss += loss.item()

                total += labels.size(0)

                if self.classification:
                    maxk = max(self.topk_acc)
                    _, pred = outputs.topk(maxk, 1, True, True)
                    pred = pred.t()
                    tmp_correct = pred.eq(labels.view(1, -1).expand_as(pred))

                    for idk, k in enumerate(self.topk_acc):
                        correct[idk] += tmp_correct[:k].reshape(-1).float().sum(0, keepdim = True)

            # Compute performance
            mean_pen = cum_pen / (i + 1)
            mean_nll = cum_nll / (i + 1)
            mean_loss = cum_loss / (i + 1)

            metrics = {'nll': mean_nll,
                       'pen': mean_pen,
                       'loss': mean_loss}

            if self.classification:
                if len(self.topk_acc) == 1:
                    mean_acc = correct[0].item() / total
                else:
                    mean_acc = [corr.item() / total for corr in correct]
                metrics['acc'] = mean_acc

            metrics = {(dsname + '_' + k): v for k, v in metrics.items()}

            return metrics

    def step_train(self):
        self.model.train()

        cum_nll = 0
        cum_pen = 0
        cum_loss = 0
        total = 0
        if self.classification:
            correct = [torch.tensor([0], dtype = self.dtype, device = self.device) for i in range(len(self.topk_acc))]
        self.idx_substep = 0
        self.logs_nlls = []
        for i, (images, labels) in enumerate(self.train_loader):
            # Convert torch tensor to Variable
            images, labels = self.loader_pre_hook(images, labels)

            # Forward + Backward + Optimize
            self.optimizer.zero_grad()    # zero the gradient buffer
            outputs = self.model(images)
            nll = self.loss_fn(outputs, labels)
            pen = torch.tensor(0.)
            loss = nll + pen

            #TODO: detailed sequence of NLLs
            self.logs_nlls.append(nll.item())

            cum_nll += nll.item()
            cum_pen += pen.item()
            cum_loss += loss.item()

            total += labels.size(0)

            if self.classification:
                maxk = max(self.topk_acc)
                _, pred = outputs.topk(maxk, 1, True, True)
                pred = pred.t()
                tmp_correct = pred.eq(labels.view(1, -1).expand_as(pred))

                for idk, k in enumerate(self.topk_acc):
                    correct[idk] += tmp_correct[:k].reshape(-1).float().sum(0, keepdim=True)

            # KFAC specific
            if self.args.optimizer.name == 'KFAC' and self.optimizer.steps % self.optimizer.TCov == 0:
                # compute true fisher
                self.optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1), 1).squeeze()
                loss_sample = self.loss_fn(outputs, sampled_y)
                loss_sample.backward(retain_graph = True)
                self.optimizer.acc_stats = False
                self.optimizer.zero_grad()  # clear the gradient for computing true-fisher.

            loss.backward()
            self.optimizer.step()


        # Compute performance
        mean_pen = cum_pen / (i + 1)
        mean_nll = cum_nll / (i + 1)
        mean_loss = cum_loss / (i + 1)

        self.model.eval()

        metrics = {'tr_nll': mean_nll,
                   'tr_pen': mean_pen,
                   'tr_loss': mean_loss}

        if self.classification:
            if len(self.topk_acc) == 1:
                mean_acc = correct[0].item() / total
            else:
                mean_acc = [corr.item() / total for corr in correct]
            metrics['tr_acc'] = mean_acc

        return metrics

    def pre_train(self):
        self.tup_params = tuple(p for n, p in self.model.named_parameters())
        self.tup_names = tuple(n for n, p in self.model.named_parameters())

        if self.args.dataset.autoencoder:
            self.loader_pre_hook = lambda x, y: loader_pre_hooks.regression(x, y, self.device, self.dtype)
        else:
            self.loader_pre_hook = lambda x, y: loader_pre_hooks.classification(x, y, self.device, self.dtype)

    def train(self, ckpt_name = 'last_ckpt', log_name = 'metrics'):
        self.build_datasets()
        self.train_loader_logs_hg = data.DataLoader(self.trainset, self.args.logs_hg.batch_size)
        self.valid_loader_logs_hg = data.DataLoader(self.validset, self.args.logs_hg.batch_size)
        self.test_loader_logs_hg = data.DataLoader(self.testset, self.args.logs_hg.batch_size)
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.model)
        self.use_scheduler = self.args.optimizer.hg.dmp_auto.use
        if self.use_scheduler and self.args.optimizer.name.find('NewtonSummary') == 0:
            args_sch = self.args.optimizer.hg.dmp_auto
            self.scheduler = ReduceDampingOnPlateau(self.optimizer, factor = args_sch.factor, 
                    patience = args_sch.patience, cooldown = args_sch.cooldown,
                    threshold = args_sch.threshold, apply_to = args_sch.apply_to, verbose = True)
        self.pre_train()

        time_t0 = time.time()

        print(self.model)
        if self.args.logs_hg.use:
            print('tup_params: ')
            for p in self.tup_params:
                print('    ', p.size())

        # Store the param names - param_groups correspondence
        torch.save(self.name_groups, f"{self.path_artifacts}/ParamNameGroups.pkl")

        # Prepare damping schedule
        damp_sch = self.args.optimizer.hg.damping_schedule
        if damp_sch != 'None':
            lst = damp_sch.split('-')
            damp_sch_init = self.args.optimizer.hg.damping
            damp_sch_final = float(lst[0])
            damp_sch_epoch = int(lst[1])
            damp_sch_factor = (damp_sch_final / damp_sch_init)**(1/(damp_sch_epoch+1))

        # Full training procedure
        for self.epoch in range(self.args.optimizer.epochs):
            print('Epoch {}'.format(self.epoch))

            # If args.logs_hg.use, then compute H, g and order3 with full-batch
            if self.args.logs_hg.use:
                logs = self.compute_logs_hg()
                torch.save(logs, f"{self.path_artifacts}/Hg_logs_ext.{self.epoch:05}.pkl")

            if self.args.logs_diff.use:
                logs = self.compute_logs_diff()
                torch.save(logs, f"{self.path_artifacts}/Hg_logs_diff.{self.epoch:05}.pkl")

            # Training step
            if self.args.optimizer.name == 'NewtonSummaryFB':
                self.model.train()
                self.optimizer.step()
                self.model.eval()

                metrics_tr = self.test_model(self.train_loader, 'tr')
            elif self.args.optimizer.name == 'LBFGS':
                def closure():
                    self.model.zero_grad()
                    objective = self.model(0)
                    objective.backward()
                    return objective

                metrics_tr = {'tr_loss': self.optimizer.step(closure).item()}
            else:
                try:
                    metrics_tr = self.step_train()
                except:
                    if self.args.optimizer.name.find("NewtonSummary") == 0 and not self.args.optimizer.hg.nologs:
                        optim_logs = self.optimizer.logs

                        logs_last = {k: v[-1] for k, v in optim_logs.items() if len(v) > 0 and torch.is_tensor(v[0])}
                        logs_mean = {k: torch.stack(v).mean(0) for k, v in optim_logs.items() if len(v) > 0 and torch.is_tensor(v[0])}
                        logs_total = {k: v for k, v in optim_logs.items() if len(v) > 0 and k != 'H'}

                        if False:
                            logs_total['H'] = optim_logs['H']

                        torch.save(logs_last, f"{self.path_artifacts}/Hg_logs_last.{self.epoch:05}.onexit.pkl")
                        torch.save(logs_mean, f"{self.path_artifacts}/Hg_logs_mean.{self.epoch:05}.onexit.pkl")
                        torch.save(logs_total, f"{self.path_artifacts}/Hg_logs_total.{self.epoch:05}.onexit.pkl")
                        torch.save(self.logs_nlls, f"{self.path_artifacts}/nlls_logs_total.{self.epoch:05}.onexit.pkl")
                    raise

            # Use scheduler
            if self.use_scheduler and self.args.optimizer.name.find('NewtonSummary') == 0:
                self.scheduler.step(metrics_tr['tr_loss'])

            metrics_va = self.test_model(self.valid_loader, 'va')
            metrics_ts = self.test_model(self.test_loader, 'ts')

            dct_time = {'epoch': self.epoch, 'time': time.time() - time_t0,
                    'memory_peak': torch.cuda.max_memory_allocated(self.device)}

            # Logs -- metrics
            metrics = dct_time | metrics_tr | metrics_va | metrics_ts
            print(metrics)

            #self.logger.log_checkpoint(self, log_name = ckpt_name)
            with open(f"{self.path_metrics}/metrics.json", "a") as f:
                f.write(metrics.__repr__() + "\n")

            # Logs -- artifacts
            if self.args.optimizer.name.find("NewtonSummary") == 0 and not self.args.optimizer.hg.nologs:
                optim_logs = self.optimizer.logs

                logs_last = {k: v[-1] for k, v in optim_logs.items() if len(v) > 0 and torch.is_tensor(v[0])}
                logs_mean = {k: torch.stack(v).mean(0) for k, v in optim_logs.items() if len(v) > 0 and torch.is_tensor(v[0])}
                logs_total = {k: v for k, v in optim_logs.items() if len(v) > 0 and k != 'H'}

                if False:
                    logs_total['H'] = optim_logs['H']

                torch.save(logs_last, f"{self.path_artifacts}/Hg_logs_last.{self.epoch:05}.pkl")
                torch.save(logs_mean, f"{self.path_artifacts}/Hg_logs_mean.{self.epoch:05}.pkl")
                torch.save(logs_total, f"{self.path_artifacts}/Hg_logs_total.{self.epoch:05}.pkl")
                torch.save(self.logs_nlls, f"{self.path_artifacts}/nlls_logs_total.{self.epoch:05}.pkl")

                self.optimizer.reset_logs() 
            elif self.args.optimizer.name == 'NewtonSummaryFB':
                torch.save(self.optimizer.logs, f"{self.path_artifacts}/Hg_logs_hgfb.{self.epoch:05}.pkl")

            # Update damping schedule
            if damp_sch != 'None' and self.epoch <= damp_sch_epoch:
                self.optimizer.damping_mul(damp_sch_factor)

        """
        metrics = add_prefix(prefix,metrics)
        print(metrics)
        self.logger.log_checkpoint(self, log_name= ckpt_name)
        self.logger.log_metrics(metrics, log_name=log_name)
        """

    def compute_logs_hg(self):
        logs = {}

        direction = fullbatch_gradient(self.param_struct, self.loss_fn, self.model, self.train_loader_logs_hg, self.train_size,
                loader_pre_hook = self.loader_pre_hook)

        H, g, order3 = compute_Hg_fullbatch(self.param_struct, self.full_loss, self.train_loader_logs_hg, self.train_size, direction, 
                loader_pre_hook = self.loader_pre_hook)
        order3_ = order3.abs().pow(1/3)

        # Compute lrs
        if not self.dct_nesterov['use']:
            regul_H = self.ridge * torch.eye(H.size(0), dtype = self.dtype, device = self.device)
            lrs = torch.linalg.solve(H + regul_H, g)
        else:
            lrs, lrs_logs = nesterov_lrs(H, g, order3_, damping_int = self.dct_nesterov['damping_int'])
            for k, v in lrs_logs.items():
                logs["nesterov." + k] = v

        logs['H'] = H
        logs['g'] = g
        logs['order3'] = order3
        logs['lrs'] = lrs

        return logs

    def compute_logs_diff(self):
        logs = {}

        if self.args.logs_diff.partition == 'canonical':
            param_groups, name_groups = build_partition.canonical(self.model)
        elif self.args.logs_diff.partition == 'wb':
            param_groups, name_groups = build_partition.wb(self.model)
        elif self.args.logs_diff.partition == 'trivial':
            param_groups, name_groups = build_partition.trivial(self.model)
        else:
            raise NotImplementedError("Not implemented: self.args.logs_diff.partition = {}".format(self.args.logs_diff.partition))

        direction = fullbatch_gradient(self.param_struct, self.loss_fn, self.model, self.train_loader_logs_hg, self.train_size,
                loader_pre_hook = self.loader_pre_hook)

        lst_diff_n = diff_n_fullbatch(self.param_struct, self.args.logs_diff.order, self.full_loss, self.train_loader_logs_hg, self.train_size, direction,
                loader_pre_hook = self.loader_pre_hook)

        logs["lst_diff_n"] = lst_diff_n

        if self.args.logs_diff.try_descent.use:
            pass

        return logs

    def lr_build_global(self, H, g):
        return g.sum() / H.sum()

    def lr_build_by_layer(self, H, g):
        #regul = self.args.optimizer.hg.epsilon * torch.eye(H.size(0), device = self.device, dtype = self.dtype)
        return torch.linalg.solve(H, g)

