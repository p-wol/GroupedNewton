import time
import copy
from collections import OrderedDict
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import mlxpy
from mlxpy import TorchModel
from kfac.optimizers import KFACOptimizer
from grnewt import compute_Hg, compute_Hg_fullbatch, fullbatch_gradient, NewtonSummary, NewtonSummaryFB
from grnewt import partition as build_partition
from grnewt.models import Perceptron, LeNet, VGG, AutoencoderMLP
from grnewt.datasets import build_MNIST, build_CIFAR10, build_toy_regression
from grnewt.nesterov import nesterov_lrs


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
        raise ValueError('Unkown dtype: {}'.format(dtype))


class Trainer:
    def __init__(self, config, logger):
        self.args = config
        self.logger= logger

        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)

        print(self.args)

        #self.build_trainer()    

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
            dct = build_MNIST(args, dct)
        elif args.dataset.name == 'CIFAR10':
            dct = build_CIFAR10(args, dct)
        elif args.dataset.name == 'ToyRegression':
            dct = build_toy_regression(args, dct)
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
            vgg_type = vgg_setup[0]
            fc_sizes = [int(s) for s in vgg_setup[1:]]

            model = VGG(vgg_type, fc_sizes = fc_sizes, image_size = self.image_size, num_classes = self.n_classes,
                        scaling = args.model.scaling, sigma_w = sigma_w, name_act_function = args.model.act_function)
        elif args.model.name == 'AutoencoderMLP':
            layers = [int(s) for s in model_args.split('-')]
            layers = [self.input_size] + layers
            classification = False

            model = AutoencoderMLP(layers, act_function, scaling = args.model.scaling, sigma_w = sigma_w, sigma_b = sigma_b)
        else:
            raise NotImplementedError('Unknown model: {}.'.format(args.model.name))

        return model.to(device = self.device, dtype = self.dtype)

    def build_optimizer(self, model):
        args = self.args
        args_hg = self.args.optimizer.hg

        # Define useful variables
        full_loss = lambda x, y: self.loss_fn(self.model(x), y)

        # Build data loader for Hg
        if args.logs_hg.use:
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
        else:
            raise NotImplementedError('Unknown partition.')

        # Build parameters for Nesterov
        dct_nesterov = None
        if args_hg.nesterov.use:
            dct_nesterov = {'use': args_hg.nesterov.use,
                            'damping_int': args_hg.nesterov.damping_int}
        self.dct_nesterov = dct_nesterov

        # Build optimizer
        if args.optimizer.name == 'SGD':
            optimizer = optim.SGD(param_groups, lr = args.optimizer.lr, 
                    momentum = args.optimizer.momentum, weight_decay = args.optimizer.weight_decay)
        elif args.optimizer.name == 'Adam':
            optimizer = optim.Adam(param_groups, lr = args.optimizer.lr)
        elif args.optimizer.name == 'NewtonSummary':
            optimizer = NewtonSummary(param_groups, full_loss, self.hg_loader, 
                    damping = args_hg.damping, momentum = args_hg.momentum, 
                    momentum_damp = args_hg.momentum_damp, period_hg = args_hg.period_hg,
                    mom_lrs = args_hg.mom_lrs, ridge = args_hg.ridge, 
                    dct_nesterov = dct_nesterov, autoencoder = args.dataset.autoencoder, 
                    remove_negative = args_hg.remove_negative)
        elif args.optimizer.name == 'NewtonSummaryFB':
            optimizer = NewtonSummaryFB(param_groups, full_loss, self.model, self.loss_fn,
                    self.hg_loader, self.train_size,
                    damping = args_hg.damping, ridge = args_hg.ridge, 
                    dct_nesterov = dct_nesterov, autoencoder = args.dataset.autoencoder)
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
        else:
            raise NotImplementedError('Unknown optimizer: {}.'.format(args.optimizer.name))

        # Store grouping data
        self.name_groups = name_groups
        self.param_groups = param_groups
        self.tup_params = tuple(p for group in self.param_groups for p in group['params'])
        self.group_sizes = [len(dct['params']) for dct in self.param_groups]
        self.group_indices = [0] + list(np.cumsum(self.group_sizes))

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
                images = images.to(device = self.device, dtype = self.dtype)
                if not self.args.dataset.autoencoder:
                    labels = labels.to(device = self.device)
                else:
                    labels = images

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
        for i, (images, labels) in enumerate(self.train_loader):
            # Convert torch tensor to Variable
            images = images.to(device = self.device, dtype = self.dtype)
            if not self.args.dataset.autoencoder:
                labels = labels.to(device = self.device)
            else:
                labels = images

            # Forward + Backward + Optimize
            self.optimizer.zero_grad()    # zero the gradient buffer
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

    def train(self, ckpt_name = 'last_ckpt', log_name = 'metrics'):
        self.build_datasets()
        self.train_loader_logs_hg = data.DataLoader(self.trainset, self.args.logs_hg.batch_size)
        self.valid_loader_logs_hg = data.DataLoader(self.validset, self.args.logs_hg.batch_size)
        self.test_loader_logs_hg = data.DataLoader(self.testset, self.args.logs_hg.batch_size)
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.model)
        self.pre_train()

        time_t0 = time.time()

        print(self.model)
        if self.args.logs_hg.use:
            print('tup_params: ')
            for p in self.tup_params:
                print('    ', p.size())

        # Store the param names - param_groups correspondence
        self.logger.log_artifact(TorchModel(self.name_groups, ext = '.pkl'), 'ParamNameGroups')

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

            # Logs Hg
            if self.args.logs_hg.use:
                if self.args.logs_hg.test_float:
                    for name_dtype, dtype in (('32', torch.float), ('64', torch.double)):
                        raise NotImplementedError('Test float/double to implement.')
                else:
                    logs = self.compute_logs()
                    self.logger.log_artifact(TorchModel(logs, ext = '.pkl'), 
                            'Hg_logs_ext.{:05}'.format(self.epoch))

            # Training step
            if self.args.optimizer.name == 'NewtonSummaryFB':
                self.model.train()
                self.optimizer.step()
                self.model.eval()

                metrics_tr = self.test_model(self.train_loader, 'tr')
            else:
                metrics_tr = self.step_train()

            metrics_va = self.test_model(self.valid_loader, 'va')
            metrics_ts = self.test_model(self.test_loader, 'ts')

            dct_time = {'epoch': self.epoch, 'time': time.time() - time_t0}

            # Logs -- metrics
            metrics = dct_time | metrics_tr | metrics_va | metrics_ts
            print(metrics)

            #self.logger.log_checkpoint(self, log_name = ckpt_name)
            self.logger.log_metrics(metrics, log_name = log_name)

            # Logs -- artifacts
            if self.args.optimizer.name == 'NewtonSummary':
                optim_logs = self.optimizer.logs

                logs_last = {k: v[-1] for k, v in optim_logs.items() if len(v) > 0}
                logs_mean = {k: torch.stack(v).mean(0) for k, v in optim_logs.items() if len(v) > 0}
                logs_total = {k: torch.stack(v) for k, v in optim_logs.items() if len(v) > 0 and k != 'H'}

                self.logger.log_artifact(TorchModel(logs_last, ext = '.pkl'),
                            'Hg_logs_last.{:05}'.format(self.epoch))
                self.logger.log_artifact(TorchModel(logs_mean, ext = '.pkl'),
                            'Hg_logs_mean.{:05}'.format(self.epoch))
                self.logger.log_artifact(TorchModel(logs_total, ext = '.pkl'),
                            'Hg_logs_total.{:05}'.format(self.epoch))

                self.optimizer.reset_logs() 
            elif self.args.optimizer.name == 'NewtonSummaryFB':
                self.logger.log_artifact(TorchModel(self.optimizer.logs, ext = '.pkl'),
                            'Hg_logs_hgfb.{:05}'.format(self.epoch))

            # Update damping schedule
            if damp_sch != 'None' and self.epoch <= damp_sch_epoch:
                self.optimizer.damping_mul(damp_sch_factor)

        """
        metrics = add_prefix(prefix,metrics)
        print(metrics)
        self.logger.log_checkpoint(self, log_name= ckpt_name)
        self.logger.log_metrics(metrics, log_name=log_name)
        """

    def compute_logs(self):
        logs = {}

        direction = fullbatch_gradient(self.model, self.loss_fn, self.tup_params, self.hg_loader, self.train_size)

        H, g, order3 = compute_Hg_fullbatch(self.tup_params, self.full_loss, self.hg_loader, self.train_size, direction, 
                param_groups = self.param_groups, group_sizes = self.group_sizes, group_indices = self.group_indices, 
                autoencoder = self.args.dataset.autoencoder)

        # Compute lrs
        if not self.dct_nesterov['use']:
            regul_H = self.ridge * torch.eye(H.size(0), dtype = self.dtype, device = self.device)
            lrs = torch.linalg.solve(H + regul_H, g)
        else:
            lrs, r_root, r_converged = nesterov_lrs(H, g, order3, 
                    damping_int = self.dct_nesterov['damping_int'])
            logs['nesterov.r'] = torch.tensor(r_root, device = self.device, dtype = self.dtype)
            logs['nesterov.converged'] = torch.tensor(r_converged, device = self.device, dtype = self.dtype)

        logs['H'] = H
        logs['g'] = g
        logs['order3'] = order3
        logs['lrs'] = lrs

        return logs

    def lr_build_global(self, H, g):
        return g.sum() / H.sum()

    def lr_build_by_layer(self, H, g):
        #regul = self.args.optimizer.hg.epsilon * torch.eye(H.size(0), device = self.device, dtype = self.dtype)
        return torch.linalg.solve(H, g)

