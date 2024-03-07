from typing import List, Dict, Any, Optional
import itertools
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from .nesterov import nesterov_lrs
from .hg import compute_Hg

class NewtonSummary(torch.optim.Optimizer):
    def __init__(self, param_groups, full_loss, data_loader: DataLoader, *,
            damping: float = 1, momentum: float = 0, momentum_damp: float = 0,
            period_hg: int = 1, mom_lrs: float = 0, ridge: float = 0, 
            dct_nesterov: dict = None, autoencoder: bool = False, noregul: bool = False,
            remove_negative: bool = False, dct_lrs_clip = None):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y_target) = l(m(x), y_target), where: 
            l: final loss (NLL, MSE...)
            m: model
            x: input
            y_target: target
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
        self.period_hg = period_hg
        self.ridge = ridge
        self.autoencoder = autoencoder
        self.noregul = noregul
        self.remove_negative = remove_negative
        defaults = {'lr': 0, 
                    'damping': damping,
                    'momentum': momentum,
                    'momentum_damp': momentum_damp,
                    'mom_lrs': mom_lrs}
        super().__init__(param_groups, defaults)

        self.tup_params = tuple(p for group in self.param_groups for p in group['params'])
        self.group_sizes = [len(dct['params']) for dct in self.param_groups]
        self.group_indices = [0] + list(np.cumsum(self.group_sizes))
        self.device = self.tup_params[0].device
        self.dtype = self.tup_params[0].dtype
        self.step_counter = 0

        if dct_nesterov is None: dct_nesterov = {'use': False}
        if 'mom_order3_' not in dct_nesterov.keys():
            dct_nesterov['mom_order3_'] = 0.
        if dct_nesterov['mom_order3_'] !=  0.:
            self.order3_ = None
        self.dct_nesterov = dct_nesterov

        if dct_lrs_clip is None: dct_lrs_clip = {'mode': 'none'}
        self.dct_lrs_clip = dct_lrs_clip

        self.reset_logs()

    def reset_logs(self):
        self.logs = {'H': [], 'g': [], 'order3': [], 'lrs': [],
                'nesterov.r': [], 'nesterov.converged': []}

    def damping_mul(self, factor):
        for group in self.param_groups:
            group['damping'] *= factor

    def _init_group(self, group: Dict[str, Any], params_with_grad: List[Tensor], 
            d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]]):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

    def step(self):
        # Update momentum buffers
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            # Update momentum buffers
            update_momentum_buffers(group['params'], d_p_list, momentum_buffer_list, 
                    momentum = group['momentum'], momentum_damp = group['momentum_damp'])

            # Update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        # Compute H, g
        if self.step_counter % self.period_hg == 0:
            # Prepare data
            x, y_target = next(self.dl_iter)
            x = x.to(device = self.device, dtype = self.dtype)
            if not self.autoencoder:
                y_target = y_target.to(device = self.device)   # XXX: dtype: this conversion has to be explicit for the user
            else:
                y_target = x
            direction = tuple(self.state[p]['momentum_buffer'] for group in self.param_groups for p in group['params'])

            # Compute H, g, order3
            H, g, order3 = compute_Hg(self.tup_params, self.full_loss, x, y_target, direction,
                    param_groups = self.param_groups, group_sizes = self.group_sizes, 
                    group_indices = self.group_indices, noregul = self.noregul)

            order3_ = order3.abs().pow(1/3)
            if self.dct_nesterov['mom_order3_'] != 0.:
                if self.order3_ is None:
                    self.order3_ = order3_
                else:
                    r = self.dct_nesterov['mom_order3_']
                    self.order3_ = r * self.order3_ + (1 - r) * order3_
                    order3_ = self.order3_

            # Compute lrs
            perform_update = True
            if self.noregul or not self.dct_nesterov['use']:
                if self.noregul:
                    regul_H = 0
                else:
                    regul_H = self.ridge * torch.eye(H.size(0), dtype = self.dtype, device = self.device)
                lrs = torch.linalg.solve(H + regul_H, g)
            else:
                lrs, lrs_logs = nesterov_lrs(H, g, order3_, 
                        damping_int = self.dct_nesterov['damping_int'])
                #print(lrs_logs)
                self.logs['nesterov.r'].append(torch.tensor(lrs_logs['r'], device = self.device, dtype = self.dtype))
                self.logs['nesterov.converged'].append(torch.tensor(lrs_logs['r_converged'], device = self.device, dtype = self.dtype))

                if not lrs_logs['found']:
                    perform_update = False
                    print('Nesterov did not converge: lr not updated during this step.')
                    #TODO: throw warning?

            # Lrs clipping
            if self.dct_lrs_clip['mode'] == 'movavg':
                #print('use clipping')
                lrs = lrs.relu()
                if not hasattr(self, 'lrs_movavg'):
                    self.lrs_movavg = lrs
                else:
                    lrs_thres = self.dct_lrs_clip['factor']
                    lrs_mom = self.dct_lrs_clip['momentum']
                    lrs_cond = self.lrs_movavg + (self.lrs_movavg == 0).to(dtype = self.dtype) * 1e9
                    if self.dct_lrs_clip['per_lr']:
                        lrs_cond = ((lrs / lrs_cond) >= lrs_thres).to(dtype = self.dtype)
                        lrs = (1 - lrs_cond) * lrs + lrs_cond * lrs_thres * self.lrs_movavg
                        #print('lrs_cond:', lrs_cond)
                    else:
                        lrs_cond_max = (lrs / lrs_cond).max()
                        #print('lrs_cond_max:', lrs_cond_max)
                        #print('lrs / lrs_cond:', lrs / lrs_cond)
                        if lrs_cond_max > lrs_thres:
                            lrs.mul_(lrs_thres / lrs_cond_max)

                    #print('lrs_movavg:', self.lrs_movavg)

                    self.lrs_movavg = lrs_mom * self.lrs_movavg + (1 - lrs_mom) * lrs
            elif self.dct_lrs_clip['mode'] == 'median':
                lrs = lrs.relu()
                if not hasattr(self, 'lrs_median'):
                    self.lrs_median = [lrs]
                else:
                    curr_lrs = lrs.clone().detach()
                    median = torch.stack(self.lrs_median).median(0).values

                    lrs_thres = self.dct_lrs_clip['factor']
                    lrs_cond = (lrs >= lrs_thres * median).to(dtype = self.dtype)

                    lrs = (1 - lrs_cond) * lrs + lrs_cond * lrs_thres * median

                    if len(self.lrs_median) == self.dct_lrs_clip['median']:
                        self.lrs_median.pop(0)
                    self.lrs_median.append(curr_lrs)
            elif self.dct_lrs_clip['mode'] == 'none':
                pass
            else:
                NotImplementedError('Error: unknown mode for lrs_clip: {}'.format(self.dct_lrs_clip['mode']))

            # Assign lrs
            if perform_update:
                for group, lr in zip(self.param_groups, lrs):
                    r = group['mom_lrs'] if self.step_counter > 0 else 0
                    lr1 = lr.item()
                    if self.remove_negative:
                        lr1 = max(0, lr1)
                    group['lr'] = r * group['lr'] + (1 - r) * group['damping'] * lr1

            # Store logs
            self.logs['H'].append(H)
            self.logs['g'].append(g)
            self.logs['order3'].append(order3)
            self.logs['lrs'].append(torch.tensor([group['lr'] for group in self.param_groups], 
                device = self.device, dtype = self.dtype))

        # Perform update
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    p.add_(state['momentum_buffer'], alpha = -group['lr'])

        self.step_counter += 1

def create_infinite_data_loader(data_loader):
    # XXX: if the batch_size does not divide the total number of samples in
    #      the data_loader, then this may fail (possibly batches of irregular sizes)
    def f():
        for dl in itertools.repeat(data_loader):
            for minibatch in dl:
                yield minibatch
    return f

def update_momentum_buffers(params: List[Tensor], d_p_list: List[Tensor], 
        momentum_buffer_list: List[Optional[Tensor]], 
        *,
        momentum: float, momentum_damp: float):
    for i, param in enumerate(params):
        d_p = d_p_list[i]

        buf = momentum_buffer_list[i]

        if buf is None:
            buf = torch.clone(d_p).detach()
            momentum_buffer_list[i] = buf
        else:
            buf.mul_(momentum).add_(d_p, alpha = 1 - momentum_damp)

