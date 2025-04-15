from typing import List, Dict, Any, Optional
import itertools
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from .nesterov import nesterov_lrs
from .hg import compute_Hg
from .util import ParamStructure

class NewtonStochasticHv(torch.optim.Optimizer):
    def __init__(self, param_groups, model, final_loss, data_loader: DataLoader, *,
            loader_pre_hook, lr_param: float, lr_direction: float,
            ridge: float = 0, dct_nesterov: dict = None):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y) = l(m(x), y), where: 
            l: final loss (NLL, MSE...)
            m: model
            x: input
            y: target
        data_loader: generate the data point for computing H and g
        lr_param: lr used in the param update
        lr_direction: lr used in the direction of descent update
        period_hg: number of training steps between each update of (H, g)
        dct_nesterov: args for Nesterov's cubic regularization procedure
            'use': True or False
            'damping_int': float; internal damping: the larger, the stronger the cubic regul.
        """
        fn_data_loader = create_infinite_data_loader(data_loader)
        self.dl_iter = iter(fn_data_loader())
        self.model = model
        self.final_loss = final_loss
        self.tup_params = tuple(model.parameters())
        self.lr_param = lr_param
        self.lr_direction = lr_direction
        self.ridge = ridge
        self.loader_pre_hook = loader_pre_hook
        self.curr_direction = None
        defaults = {'lr': 0}
        super().__init__(param_groups, defaults)

        self.param_struct = ParamStructure(param_groups)
        self.device = self.param_struct.device
        self.dtype = self.param_struct.dtype

        self.step_counter = 0

        if dct_nesterov is None: 
            dct_nesterov = {'use': False}
        if 'mom_order3_' not in dct_nesterov.keys():
            dct_nesterov['mom_order3_'] = 0.
        if dct_nesterov['mom_order3_'] != 0.:
            self.order3_ = None
        self.dct_nesterov = dct_nesterov

        self.reset_logs()

    def reset_logs(self):
        if hasattr(self, 'logs'):
            del self.logs
        self.logs = {'H': [], 'g': [], 'order3': [], 'lrs': [], 'lrs_clipped': [],
                'curr_lrs': [], 'nesterov.r': [], 'nesterov.converged': []}

    def damping_mul(self, factor):
        for group in self.param_groups:
            group['damping'] *= factor
            group['lr'] *= factor

    def step(self):
        # Compute the gradient
        grad = tuple(p.grad for p in self.tup_params)
        if self.curr_direction is None:
            self.curr_direction = grad

        # Prepare data
        x, y = next(self.dl_iter)
        x, y = self.loader_pre_hook(x, y)

        # Full loss function w.r.t. the parameters
        def full_loss(*params):
            output = torch.func.functional_call(self.model, {k: p for (k, v), p in zip(self.model.named_parameters(), params)}, x)
            return self.final_loss(output, y)

        # Compute Hessian-vector product
        _, vhp = torch.autograd.functional.vhp(full_loss, self.tup_params, v = self.curr_direction)

        # Update direction: d_{t+1} = d_t - lr_direction * (H d_t - grad)
        with torch.no_grad():
            for d, g, v in zip(self.curr_direction, grad, vhp):
                if self.ridge == 0:
                    d.add_(v - g, alpha = -self.lr_direction)
                else:
                    d.add_(v - g + self.ridge * d, alpha = -self.lr_direction)

        # Update params
        with torch.no_grad():
            for p, d in zip(self.tup_params, self.curr_direction):
                p.add_(d, alpha = -self.lr_param)

        # Store logs
        """
        self.logs['H'].append(H)
        self.logs['g'].append(g)
        self.logs['order3'].append(order3)
        self.logs['lrs'].append(torch.tensor([group['lr'] for group in self.param_groups], 
            device = self.device, dtype = self.dtype))
        """

        # Perform update
        """
        with torch.no_grad():
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.add_(direction[i], alpha = -group['lr'])
                    i += 1
        """

        self.step_counter += 1

def create_infinite_data_loader(data_loader):
    # XXX: if the batch_size does not divide the total number of samples in
    #      the data_loader, then this may fail (possibly batches of irregular sizes)
    def f():
        for dl in itertools.repeat(data_loader):
            for minibatch in dl:
                yield minibatch
    return f

