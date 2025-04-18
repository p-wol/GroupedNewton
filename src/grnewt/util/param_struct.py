import itertools
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

# XXX: to test
# XXX: check if every variable V requires grad when used in autograd.grad(V, ...)


class ParamStructure:
    def __init__(self, pgroups):
        self.pgroups = pgroups
        self.nb_groups = len(self.pgroups)
        self.tup_params = tuple(p for group in self.pgroups for p in group['params'])
        self.group_sizes = [len(dct['params']) for dct in self.pgroups]
        self.group_indices = [0] + list(np.cumsum(self.group_sizes))
        self.device = self.tup_params[0].device
        self.dtype = self.tup_params[0].dtype

    def select_params(self, end = None, src = None, *, start):
        end = end if end is not None else self.nb_groups

        if src is None:
            return tuple(p for group in self.pgroups[start:end] for p in group['params'])
        else:
            p_start = self.group_indices[start]
            p_end = self.group_indices[end]
            return src[p_start:p_end]

    def dot(self, x1, x2, *, dst_type = "tensor", start = 0, end = None):
        end = end if end is not None else self.nb_groups

        i0 = self.group_indices[start]
        pdot = [(p1 * p2).sum() for p1, p2 in zip(x1, x2)]
        res = [sum(pdot[i1-i0:i2-i0]) for i1, i2 in zip(self.group_indices[start:end], self.group_indices[start+1:end+1])]
        if dst_type == "list":
            return res
        elif dst_type == "tuple":
            return tuple(res)
        elif dst_type == "tensor":
            #res = [p if torch.is_tensor(p) else torch.tensor(p, device = self.device, dtype = self.dtype) for p in res]
            return torch.stack(res)
        else:
            raise NotImplementedError(f"Unknown dct_type: {dst_type}.")

    def regroup(self, x, *, dst_type = "tensor", start = 0, end = None):
        end = end if end is not None else self.nb_groups

        i0 = self.group_indices[start]
        pdot = [x_.sum() for x_ in x]
        res = [sum(pdot[i1-i0:i2-i0]) for i1, i2 in zip(self.group_indices[start:end], self.group_indices[start+1:end+1])]
        if dst_type == "list":
            return res
        elif dst_type == "tuple":
            return tuple(res)
        elif dst_type == "tensor":
            return torch.stack(res)
        else:
            raise NotImplementedError(f"Unknown dct_type: {dst_type}.")

    def dercon(self, gpar, gdir, start, end, *, detach):
        # Returns zero tensors if gpar does not require grad
        if not gpar.requires_grad:
            end = end if end is not None else self.nb_groups
            return torch.zeros(end - start, device = self.device, dtype = self.dtype)

        # Derivation + contraction
        if detach:
            kwargs = {"retain_graph": True, "materialize_grads": True}
        else:
            kwargs = {"create_graph": True, "materialize_grads": True}

        tup_params = self.select_params(start = start, end = end)
        deriv = torch.autograd.grad(gpar, tup_params, **kwargs)

        direction = self.select_params(src = gdir, start = start, end = end)
        return self.dot(deriv, direction, start = start, end = end)

    def vhpcon(self, func, direction, start, end, *, detach):
        # Derivation + contraction
        tup_params = self.select_params(start = start, end = end)
        direction = self.select_params(src = direction, start = start, end = end)

        _, vhp = torch.autograd.functional.vhp(func, tup_params, v = direction, create_graph = not detach)

        return self.regroup(vhp, start = start, end = end)

