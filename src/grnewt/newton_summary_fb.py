from typing import List, Dict, Any, Optional
import itertools
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from .nesterov import nesterov_lrs
from .hg import compute_Hg, compute_Hg_fullbatch
from .util import fullbatch_gradient

class NewtonSummaryFB(torch.optim.Optimizer):
    def __init__(self, params, full_loss, model, final_loss, data_loader: DataLoader, dataset_size: int, *,
            loader_pre_hook, damping: float = 1, ridge: float = 0, 
            dct_nesterov: dict = None, noregul: bool = False,
            remove_negative: bool = False):
        """
        params: params of the model
        full_loss: full_loss(x, y_target) = l(m(x), y_target), where: 
            l: final loss (NLL, MSE...)
            m: model
            x: input
            y_target: target
        model: model to train
        final_loss: final loss (NLL, MSE...)   
        data_loader: generate the data point for computing H and g
        damping: "damping" as in Newton's method (can be seen as a correction of the lr)
        """
        # XXX: "final_loss" should have reduction = 'mean'
        self.model = model
        self.final_loss = final_loss
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.full_loss = full_loss
        self.ridge = ridge
        self.loader_pre_hook = loader_pre_hook
        self.noregul = noregul
        self.remove_negative = remove_negative
        defaults = {'lr': 0, 
                    'damping': damping}
        super().__init__(params, defaults)

        self.tup_params = tuple(p for group in self.param_groups for p in group['params'])
        self.group_sizes = [len(dct['params']) for dct in self.param_groups]
        self.group_indices = [0] + list(np.cumsum(self.group_sizes))
        self.device = self.tup_params[0].device
        self.dtype = self.tup_params[0].dtype
        self.step_counter = 0

        if dct_nesterov is None: dct_nesterov = {'use': False}
        self.dct_nesterov = dct_nesterov

        self.logs = {}

    def get_lrs(self):
        return [group['lr'] for group in self.param_groups]

    def damping_mul(self, factor):
        for group in self.param_groups:
            group['damping'] *= factor

    def _init_group(self, group: Dict[str, Any], params_with_grad: List[Tensor], 
            d_p_list: List[Tensor]):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

    def step(self):
        # Update groups
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            self._init_group(group, params_with_grad, d_p_list)

        # Compute lrs when using the fullbatch gradient direction
        direction = fullbatch_gradient(self.model, self.final_loss, self.tup_params, self.data_loader, self.dataset_size, 
                loader_pre_hook = self.loader_pre_hook)

        lrs = self.compute_lrs(direction, nesterov_damping = self.dct_nesterov['damping_int'], 
                noregul = self.noregul)

        # Assign lrs
        for group, lr in zip(self.param_groups, lrs):
            lr1 = lr.item()
            if self.remove_negative:
                lr1 = max(0, lr1)
            group['lr'] = group['damping'] * lr1


        # Perform update
        with torch.no_grad():
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.add_(direction[i], alpha = -group['lr'])
                    i += 1
        self.step_counter += 1

    def compute_lrs(self, direction, *, nesterov_damping = None, noregul = False):
        tup_params = self.tup_params

        H, g, order3 = compute_Hg_fullbatch(self.tup_params, self.full_loss, self.data_loader,
                self.dataset_size, direction, param_groups = self.param_groups, 
                group_sizes = self.group_sizes, group_indices = self.group_indices,
                noregul = self.noregul, loader_pre_hook = self.loader_pre_hook)

        # Use Nesterov cubic regularization (if necessary)
        if noregul:
            lrs = torch.linalg.solve(H, g)
            self.logs['H'] = H
            self.logs['g'] = g
            self.logs['lrs'] = torch.tensor([group['lr'] for group in self.param_groups], 
                device = self.device, dtype = self.dtype)
        else:
            lrs, r_root, r_converged = nesterov_lrs(H, g, order3, 
                    damping_int = nesterov_damping)

            # Store logs
            self.logs['H'] = H
            self.logs['g'] = g
            self.logs['lrs'] = torch.tensor([group['lr'] for group in self.param_groups], 
                device = self.device, dtype = self.dtype)
            self.logs['order3'] = order3
            self.logs['nesterov.r'] = torch.tensor(r_root, device = self.device, dtype = self.dtype)
            self.logs['nesterov.converged'] = torch.tensor(r_converged, device = self.device, dtype = self.dtype)

        return lrs
