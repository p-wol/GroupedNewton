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
    def __init__(self, param_groups, full_loss, data_loader: DataLoader, updater, *,
            loader_pre_hook,
            damping: float = 1, period_hg: int = 1, mom_lrs: float = 0, movavg: float = 0, ridge: float = 0, 
            dct_nesterov: dict = None, noregul: bool = False,
            remove_negative: bool = False, maintain_true_lrs = False,
            diagonal = False):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y) = l(m(x), y), where: 
            l: final loss (NLL, MSE...)
            m: model
            x: input
            y: target
        data_loader: generate the data point for computing H and g
        damping: "damping" as in Newton's method (can be seen as a correction of the lr)
        momentum: "momentum" as in SGD
        momentum_damp: "dampening" of the momentum as in SGD
        period_hg: number of training steps between each update of (H, g)
        mom_lrs: momentum for the updates of lrs
        dct_nesterov: args for Nesterov's cubic regularization procedure
            'use': True or False
            'damping_int': float; internal damping: the larger, the stronger the cubic regul.
        """
        self.fn_data_loader = create_infinite_data_loader(data_loader)
        self.dl_iter = iter(self.fn_data_loader())
        self.full_loss = full_loss
        self.updater = updater
        self.period_hg = period_hg
        self.ridge = ridge
        self.loader_pre_hook = loader_pre_hook
        self.noregul = noregul
        self.remove_negative = remove_negative
        self.mom_lrs = mom_lrs
        self.movavg = movavg
        self.maintain_true_lrs = maintain_true_lrs
        self.curr_lrs = 0
        self.diagonal = diagonal
        defaults = {'lr': 0, 
                    'damping': damping}
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

        if self.movavg != 0:
            self.H = None
            self.g = None
            self.order3 = None

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
        direction = self.updater.compute_step()

        # Compute H, g
        perform_update = True
        if self.step_counter % self.period_hg == 0:
            # Prepare data
            x, y = next(self.dl_iter)
            x, y = self.loader_pre_hook(x, y)

            # TODO: step

            #if self.diagonal:
            #    H = H.diag().diag()


            # Nesterov cubic regul?


            # Assign lrs

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

