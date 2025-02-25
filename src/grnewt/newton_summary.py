from typing import List, Dict, Any, Optional
import itertools
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from .nesterov import nesterov_lrs
from .hg import compute_Hg
from .util import ParamStructure

class NewtonSummary(torch.optim.Optimizer):
    def __init__(self, param_groups, full_loss, data_loader: DataLoader, updater, *,
            loader_pre_hook,
            damping: float = 1, period_hg: int = 1, mom_lrs: float = 0, movavg: float = 0, ridge: float = 0, 
            dct_nesterov: dict = None, noregul: bool = False,
            remove_negative: bool = False, dct_lrs_clip = None, maintain_true_lrs = False,
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

        if dct_lrs_clip is None: 
            dct_lrs_clip = {'mode': 'none'}
        self.dct_lrs_clip = dct_lrs_clip

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

    """
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
    """

    def step(self):
        direction = self.updater.compute_step()

        # Compute H, g
        perform_update = True
        if self.step_counter % self.period_hg == 0:
            # Prepare data
            x, y = next(self.dl_iter)
            x, y = self.loader_pre_hook(x, y)

            # Compute H, g, order3
            H, g, order3 = compute_Hg(self.param_struct, self.full_loss, x, y, direction,
                    noregul = self.noregul, diagonal = self.diagonal)

            #if self.diagonal:
            #    H = H.diag().diag()

            order3_ = order3.abs().pow(1/3)
            if self.dct_nesterov['mom_order3_'] != 0.:
                if self.order3_ is None:
                    self.order3_ = order3_
                else:
                    r = self.dct_nesterov['mom_order3_']
                    self.order3_ = r * self.order3_ + (1 - r) * order3_
                    order3_ = self.order3_

            if self.movavg != 0:
                if self.H is None:
                    self.H = H
                    self.g = g
                    self.order3 = order3
                else:
                    r = self.movavg
                    self.H = r * self.H + (1 - r) * H
                    self.g = r * self.g + (1 - r) * g
                    self.order3 = r * self.order3 + (1 - r) * order3

                    H = self.H
                    g = self.g
                    order3 = self.order3
                order3_ = order3_ = order3.abs().pow(1/3)

            # Compute lrs
            if self.noregul or not self.dct_nesterov['use']:
                if self.noregul:
                    regul_H = 0
                else:
                    regul_H = self.ridge * torch.eye(H.size(0), dtype = self.dtype, device = self.device)
                lrs = torch.linalg.solve(H + regul_H, g)
            else:
                if self.dct_lrs_clip['mode'] != 'median_r' or not hasattr(self, 'r_median'):
                    clip_r = None
                else:
                    median = torch.stack(self.r_median).median(0).values
                    clip_r = self.dct_lrs_clip['factor'] * median

                lrs, lrs_logs = nesterov_lrs(H, g, order3_, 
                        damping_int = self.dct_nesterov['damping_int'], clip_r = clip_r)

                for k, v in lrs_logs.items():
                    kk = 'nesterov.' + k
                    if kk not in self.logs.keys():
                        self.logs[kk] = []
                    self.logs[kk].append(v)

                if not lrs_logs['found']:
                    perform_update = False
                    print('Nesterov did not converge: lr not updated during this step.')
                    #TODO: throw warning?

            if not perform_update:
                lrs = torch.zeros(g.size(0), dtype = self.dtype, device = self.device)
            else:
                # Lrs clipping
                if self.dct_lrs_clip['mode'] == 'movavg':
                    if not hasattr(self, 'lrs_movavg'):
                        self.lrs_movavg = lrs
                    else:
                        lrs_thres = self.dct_lrs_clip['factor']
                        lrs_mom = self.dct_lrs_clip['momentum']
                        lrs_movavg = self.lrs_movavg.relu()
                        lrs_cond = lrs_movavg + (lrs_movavg == 0).to(dtype = self.dtype) * 1e9
                        if self.dct_lrs_clip['per_lr']:
                            lrs_cond = (lrs >= lrs_thres * lrs_cond).to(dtype = self.dtype)
                            lrs = (1 - lrs_cond) * lrs + lrs_cond * lrs_thres * self.lrs_movavg
                        else:
                            lrs_cond_max = (lrs / lrs_cond).max()
                            if lrs_cond_max > lrs_thres:
                                lrs.mul_(lrs_thres / lrs_cond_max)

                        self.lrs_movavg = lrs_mom * self.lrs_movavg + (1 - lrs_mom) * lrs
                elif self.dct_lrs_clip['mode'] == 'median':
                    #lrs = lrs.relu()
                    if not hasattr(self, 'lrs_median'):
                        self.lrs_median = [lrs]
                    else:
                        curr_lrs = lrs.clone().detach()
                        median = torch.stack(self.lrs_median).median(0).values.relu()

                        lrs_thres = self.dct_lrs_clip['factor']
                        lrs_cond = (lrs >= lrs_thres * median).to(dtype = self.dtype)

                        lrs = (1 - lrs_cond) * lrs + lrs_cond * lrs_thres * median

                        if len(self.lrs_median) == self.dct_lrs_clip['median']:
                            self.lrs_median.pop(0)
                        if self.dct_lrs_clip['incremental']:
                            self.lrs_median.append(lrs)
                        else:
                            self.lrs_median.append(curr_lrs)
                elif self.dct_lrs_clip['mode'] == 'median_r':
                    curr_r = lrs_logs['r']
                    if not hasattr(self, 'r_median'):
                        self.r_median = []
                    self.r_median.append(curr_r)
                    if len(self.r_median) > self.dct_lrs_clip['median']:
                        self.r_median.pop(0)
                elif self.dct_lrs_clip['mode'] == 'none':
                    pass
                else:
                    NotImplementedError('Error: unknown mode for lrs_clip: {}'.format(self.dct_lrs_clip['mode']))

            # To execute even when update_lrs = False? Block #1
            r = self.mom_lrs if self.step_counter > 0 else 0
            if self.maintain_true_lrs:
                self.curr_lrs = r * self.curr_lrs + (1 - r) * lrs
                lrs = self.curr_lrs
                if self.remove_negative:
                    lrs = lrs.relu()
            else:
                if self.remove_negative:
                    lrs = lrs.relu()
                self.curr_lrs = r * self.curr_lrs + (1 - r) * lrs
                lrs = self.curr_lrs

            # To execute even when update_lrs = False? Block #2
            # Assign lrs
            self.logs['lrs_clipped'].append(lrs)
            self.logs['curr_lrs'].append(self.curr_lrs)
            for group, lr in zip(self.param_groups, lrs):
                group['lr'] = group['damping'] * lr.item()

            # Store logs
            self.logs['H'].append(H)
            self.logs['g'].append(g)
            self.logs['order3'].append(order3)
            self.logs['lrs'].append(torch.tensor([group['lr'] for group in self.param_groups], 
                device = self.device, dtype = self.dtype))

        # Perform update
        with torch.no_grad():
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.add_(direction[i], alpha = -group['lr'])
                    i += 1

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

