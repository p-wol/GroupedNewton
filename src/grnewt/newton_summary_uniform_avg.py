from typing import List, Dict, Any, Optional
import itertools
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from .nesterov import nesterov_lrs
from .hg import compute_Hg
from .util import ParamStructure

class NewtonSummaryUniformAvg(torch.optim.Optimizer):
    def __init__(self, param_groups, full_loss, data_loader: DataLoader, updater, *,
            loader_pre_hook,
            damping: float = 1, period_hg: int = 1, mom_lrs: float = 0, ridge: float = 0, 
            dct_nesterov: dict = None, noregul: bool = False,
            remove_negative: bool = False, dct_uniform_avg = None):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y) = l(m(x), y), where: 
            l: final loss (NLL, MSE...)
            m: model
            x: input
            y: target
        data_loader: generate the data point for computing H and g
        damping: "damping" as in Newton's method (can be seen as a correction of the lr)
        period_hg: number of training steps between each update of (H, g)
        mom_lrs: momentum for the updates of lrs
        dct_nesterov: args for Nesterov's cubic regularization procedure
            'use': True or False
            'damping_int': float; internal damping: the larger, the stronger the cubic regul.
        dct_uniform_avg: args for uniform average
            Idea: 
                update H, g, D in the following way:
                    X_{t+1} = (t/(t+1))*X_t + (1/(t+1))*x_t
                for each variable to maintain, update two elements:
                    X^a and X^b:
                    beginning: X^a is well-defined, X^b = x_0
                    for T steps, update and use X^a, and update X^b
                    at the end: X_a <- X_b; X_b <- x_T
                    and repeat.
            'use': True or False
            'period': int; number of Hg update steps between each change of X^a, X^b.
            'warmup': int; number of Hg update steps during which H, g, D are updated, but the NN is *not trained*
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
        self.curr_lrs = 0
        defaults = {'lr': 0, 
                    'damping': damping}
        super().__init__(param_groups, defaults)

        self.param_struct = ParamStructure(param_groups)
        self.device = self.param_struct.device
        self.dtype = self.param_struct.dtype
        self.step_counter = 0

        # Init dct_nesterov
        if dct_nesterov is None: 
            dct_nesterov = {'use': False}
        if 'mom_order3_' not in dct_nesterov.keys():
            dct_nesterov['mom_order3_'] = 0.
        if dct_nesterov['mom_order3_'] !=  0.:
            self.order3_ = None
        self.dct_nesterov = dct_nesterov

        # Init dct_uniform_avg
        if dct_uniform_avg is None:
            raise NotImplementedError("TODO: handle case of 'dct_uniform_avg is None'.")

        self.warmup = dct_uniform_avg["warmup"]
        self.unif_avg_period = dct_uniform_avg["period"]
        self.dct_HgD_avgs = {k: None for k in ["H_use", "H_up", "g_use", "g_up", "D_use", "D_up"]}

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

    def update_uniform_avg(self, H, g, order3):
        """
        Updates the attribute dct_HgD_avgs.
        """
        dct_HgD = {"H": H, "g": g, "D": order3}

        # Compute the time step of the method "uniform average"
        t = (self.step_counter // self.period_hg) % self.unif_avg_period

        # Time to replace the current moving average
        if t == 0:
            # At init, set the moving averages to zero
            if self.step_counter == 0:
                for key, curr in dct_HgD.items():
                    self.dct_HgD_avgs[f"{key}_up"] = curr

            # Replace the current moving average by the next one and set the other to zero
            for key, curr in dct_HgD.items():
                self.dct_HgD_avgs[f"{key}_use"] = self.dct_HgD_avgs[f"{key}_up"]
                self.dct_HgD_avgs[f"{key}_up"] = curr

        # During the first period, both averages "use" and "up" are updated with the same coefficient
        offset_use = 0 if self.step_counter // self.period_hg < self.unif_avg_period else self.unif_avg_period
        
        # Set up the time coefficients
        tt_use = offset_use + t + 1
        tt_up = t + 1

        # Update the moving averages
        for key1, curr in dct_HgD.items():
            for key2, n in zip(["use", "up"], [tt_use, tt_up]):
                key = f"{key1}_{key2}"
                self.dct_HgD_avgs[key].mul_((n - 1)/n).add_((1/n) * curr)

        # Return the H, g, order3 to use
        H = self.dct_HgD_avgs["H_use"]
        g = self.dct_HgD_avgs["g_use"]
        order3 = self.dct_HgD_avgs["D_use"]

        return H, g, order3

    def step(self):
        # Perform update only if warm-up phase has ended
        perform_update = (self.step_counter // self.period_hg >= self.warmup)

        # Compute the direction of descent
        direction = self.updater.compute_step()

        # Function that performs an update if necessary
        def move_forward():
            self.step_counter += 1

            if not perform_update:
                return
            
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    for p in group['params']:
                        p.add_(direction[i], alpha = -group['lr'])
                        i += 1

        # If we do not update H, g, order3 and lrs: just move forward
        if self.step_counter % self.period_hg != 0:
            move_forward()
            return

        # Compute H, g
        ## Prepare data
        x, y = next(self.dl_iter)
        x, y = self.loader_pre_hook(x, y)

        ## Compute H, g, order3
        H, g, order3 = compute_Hg(self.param_struct, self.full_loss, x, y, direction,
                noregul = self.noregul, diagonal = False)

        # Update the averages of H, g, order3
        H, g, order3 = self.update_uniform_avg(H, g, order3)

        order3_ = order3.abs().pow(1/3)

        ## Store logs of H, g, order3
        self.logs['H'].append(H)
        self.logs['g'].append(g)
        self.logs['order3'].append(order3)

        # If we are still in the warm-up phase, just move forward
        if not perform_update:
            move_forward()
            return

        # Compute lrs
        if self.noregul or not self.dct_nesterov['use']:
            if self.noregul:
                regul_H = 0
            else:
                regul_H = self.ridge * torch.eye(H.size(0), dtype = self.dtype, device = self.device)
            lrs = torch.linalg.solve(H + regul_H, g)
        else:
            lrs, lrs_logs = nesterov_lrs(H, g, order3_, damping_int = self.dct_nesterov['damping_int'])

            for k, v in lrs_logs.items():
                kk = 'nesterov.' + k
                if kk not in self.logs.keys():
                    self.logs[kk] = []
                self.logs[kk].append(v)

            if not lrs_logs['found']:
                perform_update = False
                print('Nesterov did not converge: lr not updated during this step.')
                #TODO: throw warning?

        ## Additional operations on the lrs
        r = self.mom_lrs if self.step_counter > 0 else 0
        self.curr_lrs = r * self.curr_lrs + (1 - r) * lrs
        lrs = self.curr_lrs
        if self.remove_negative:
            lrs = lrs.relu()

        ## Assign lrs
        self.logs['lrs_clipped'].append(lrs)
        self.logs['curr_lrs'].append(self.curr_lrs)
        for group, lr in zip(self.param_groups, lrs):
            group['lr'] = group['damping'] * lr.item()

        # Store logs of lrs
        self.logs['lrs'].append(torch.tensor([group['lr'] for group in self.param_groups], 
            device = self.device, dtype = self.dtype))

        # Perform update
        move_forward()

def create_infinite_data_loader(data_loader):
    # XXX: if the batch_size does not divide the total number of samples in
    #      the data_loader, then this may fail (possibly batches of irregular sizes)
    def f():
        for dl in itertools.repeat(data_loader):
            for minibatch in dl:
                yield minibatch
    return f

