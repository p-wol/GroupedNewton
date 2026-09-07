import numpy as np
import torch

# XXX: to test
# XXX: check if every variable V requires grad when used in autograd.grad(V, ...)


class ParamStructure:
    def __init__(self, pgroups):
        self.pgroups = pgroups
        self.nb_groups = len(self.pgroups)
        self.tup_params = tuple(p for group in self.pgroups for p in group["params"])
        self.group_sizes = [len(dct["params"]) for dct in self.pgroups]
        self.group_indices = [0] + list(np.cumsum(self.group_sizes))
        self.device = self.tup_params[0].device
        self.dtype = self.tup_params[0].dtype

    def select_params(self, end=None, src=None, *, start):
        end = end if end is not None else self.nb_groups

        if src is None:
            return tuple(p for group in self.pgroups[start:end] for p in group["params"])
        else:
            p_start = self.group_indices[start]
            p_end = self.group_indices[end]
            return src[p_start:p_end]

    def build_reindex(self, src_params):
        """Permutation taking a tuple indexed like `src_params` to tup_params order.

        Necessary, not cosmetic.  `compute_Hg` and every `p.add_(direction[i])`
        loop index `direction` in *tup_params* order, i.e. in *partition* order.
        The updater that produces `direction` is built from `model.parameters()`,
        which is a different order for every partition that regroups tensors
        (`wb`, `blocks-k`, `alternate-k`, and the `vgg`/`perceptron` builders).
        Feeding the wrong order does not raise: `ParamStructure.dot` contracts
        with `(p1 * p2).sum()`, which BROADCASTS, so Hbar and gbar come out
        silently wrong and the run only crashes later -- if the shapes happen to
        be incompatible for the in-place `add_`.

        Returns None when the two orders already coincide, so the common
        `canonical` / `trivial` case costs nothing.
        """
        pos = {}
        for i, p in enumerate(src_params):
            pos.setdefault(id(p), i)
        missing = [p for p in self.tup_params if id(p) not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} parameter(s) of the partition are absent from the "
                "producer of `direction`; the partition and the updater were built "
                "from different parameter sets."
            )
        perm = [pos[id(p)] for p in self.tup_params]
        return None if perm == list(range(len(perm))) else perm

    def reindex(self, values, perm):
        """Apply `build_reindex`'s permutation, checking shapes."""
        if perm is None:
            out = tuple(values)
        else:
            out = tuple(values[j] for j in perm)
        for v, p in zip(out, self.tup_params, strict=False):
            if v.shape != p.shape:
                raise ValueError(
                    f"direction entry of shape {tuple(v.shape)} paired with a "
                    f"parameter of shape {tuple(p.shape)}"
                )
        return out

    def dot(self, x1, x2, *, dst_type="tensor", start=0, end=None):
        end = end if end is not None else self.nb_groups

        i0 = self.group_indices[start]
        pdot = [(p1 * p2).sum() for p1, p2 in zip(x1, x2, strict=False)]
        res = [
            sum(pdot[i1 - i0 : i2 - i0])
            for i1, i2 in zip(
                self.group_indices[start:end], self.group_indices[start + 1 : end + 1], strict=False
            )
        ]
        if dst_type == "list":
            return res
        elif dst_type == "tuple":
            return tuple(res)
        elif dst_type == "tensor":
            # res = [p if torch.is_tensor(p) else torch.tensor(p, device = self.device, dtype = self.dtype) for p in res]
            return torch.stack(res)
        else:
            raise NotImplementedError(f"Unknown dct_type: {dst_type}.")

    def squared_norm(self, x):
        return self.dot(x, x)

    def expand_src_as_params(self, src):
        """
        For an input src = (t1, t2, ..., tS),
        build a tuple (t1, t1, t1, t2, t2, ..., tS), where each ts is duplicated
        ns times, where ns is the number of tensor parameters in group s.
        """
        lst_groups = [[t]*len(group["params"]) for t, group in zip(src, self.pgroups, strict=True)]

        return tuple(t for g in lst_groups for t in g)

    def dercon(self, gpar, gdir, start, end, *, detach):
        # Returns zero tensors if gpar does not require grad
        if not gpar.requires_grad:
            end = end if end is not None else self.nb_groups
            return torch.zeros(end - start, device=self.device, dtype=self.dtype)

        # Derivation + contraction
        if detach:
            kwargs = {"retain_graph": True, "materialize_grads": True}
        else:
            kwargs = {"create_graph": True, "materialize_grads": True}

        tup_params = self.select_params(start=start, end=end)
        deriv = torch.autograd.grad(gpar, tup_params, **kwargs)

        direction = self.select_params(src=gdir, start=start, end=end)
        return self.dot(deriv, direction, start=start, end=end)
