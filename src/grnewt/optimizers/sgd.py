from typing import List, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class SGDUpdate(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    def _init_group(self, group, params, grads, momentum_buffer_list, updates):
        has_sparse_grad = False

        for p in group["params"]:
            params.append(p)
            grads.append(p.grad)
            if p.grad.is_sparse:
                has_sparse_grad = True

            state = self.state[p]
            if group["momentum"] != 0:
                momentum_buffer_list.append(state.get("momentum_buffer"))

            updates.append(state.get("update"))
            #state["update"] = torch.zeros_like(
            #    p, memory_format=torch.preserve_format
            #)
            #updates.append(state["update"])

        return has_sparse_grad

    def compute_step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lst_updates = []
        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []
            updates: List[Tensor] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list, updates
            )

            sgd(
                params,
                grads,
                momentum_buffer_list,
                updates,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

            for p, update in zip(params, updates):
                state = self.state[p]
                state["update"] = update

            lst_updates += updates

        return tuple(lst_updates)
        
    def step(self, tup_updates):
        with torch.no_grad():
            j = 0
            for group in self.param_groups:
                for i, param in enumerate(group["params"]):
                    param.add_(tup_updates[j])
                    j += 1


def sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    updates: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        if updates[i] is None:
            updates[i] = torch.zeros_like(param, memory_format = torch.preserve_format)

        updates[i].zero_().add_(grad, alpha=lr)
