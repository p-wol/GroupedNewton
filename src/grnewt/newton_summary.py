import torch

from .config import HgCfg
from .hg import compute_Hg, compute_Hg_batched
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure


def increment_step(func):
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.step_counter += 1
        return ret

    return wrapper


class NewtonSummary(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        loss_fn,
        updater,
        *,
        cfg: HgCfg,
    ):
        """
        param_groups: param_groups of the model
        loss_fn: CALLABLE of no argument returning the scalar objective at the CURRENT
            parameters, with its autograd graph (i.e. `lambda: f(theta)`, not `f(theta)`).

            `step()` calls it once and uses the same graph for the gradient (which feeds
            the updater) and for (Hbar, gbar, order3), so the forward pass is not paid
            twice.
        updater: produces the candidate direction u from `.grad`. `step()` fills `.grad`
            itself, so the caller must NOT call `loss.backward()` beforehand.
        cfg: validated `optimizer.hg` node; see grnewt/config.py for every field,
             its default, and which optimizers read it. Nothing else is read from
             the config, and every field this optimizer ignores is rejected at
             composition time by grnewt.config.check_consumed.

        Unlike the NSBase family this optimizer has no data loader and no averaging:
        one call to `step()` is one exact evaluation of the summaries at the current
        point. `mom_lrs` and `maintain_true_lrs` are therefore not read (config.py
        excludes NewtonSummary from their `used_by`), and a failure of `nesterov_lrs`
        raises instead of holding the previous learning rates.
        """
        if torch.is_tensor(loss_fn):
            raise TypeError(
                "NewtonSummary takes a callable returning the loss, not a loss tensor: "
                "pass `lambda: f(theta)`, not `f(theta)`."
            )
        if not callable(loss_fn):
            raise TypeError(f"loss_fn must be callable, got {type(loss_fn).__name__}")
        self.loss_fn = loss_fn
        self.updater = updater
        self.cfg = cfg
        self.curr_lrs = 0
        self.dir_norm = None

        # `damping` is per-parameter-group state, not a hyperparameter of the run:
        # damping_mul() mutates it. It therefore belongs to torch's `defaults`
        # mechanism, not to `self.cfg`. Every other field stays in `self.cfg`.
        super().__init__(param_groups, {"lr": 0, "damping": cfg.damping})

        self.param_struct = ParamStructure(param_groups)
        self.device = self.param_struct.device
        self.dtype = self.param_struct.dtype

        # See ParamStructure.build_reindex.
        self._dir_perm = self.param_struct.build_reindex(
            [p for gr in updater.param_groups for p in gr["params"]]
        )

        self.step_counter = 0

        self.reset_logs()

    def reset_logs(self):
        if hasattr(self, "logs"):
            del self.logs
        self.logs = {
            "H": [],
            "g": [],
            "order3": [],
            "lrs": [],
            "lrs_clipped": [],
            "curr_lrs": [],
            "nesterov.r": [],
            "nesterov.converged": [],
            "loss": [],
        }

    def damping_mul(self, factor):
        for group in self.param_groups:
            group["damping"] *= factor
            group["lr"] *= factor

    def normalize_dirs_(self, direction):
        norm = self.param_struct.squared_norm(direction).sqrt()
        norm = self.param_struct.expand_src_as_params(norm)

        for d, n in zip(direction, norm, strict=True):
            if n > 0:
                d.div_(n)

        return norm

    def compute_Hg(self, loss, direction):
        cp_kwargs = {"noregul": self.cfg.noregul, "diagonal": self.cfg.diagonal}
        if self.cfg.hg_batched:
            cp_Hg = compute_Hg_batched
            cp_kwargs["chunk_size"] = self.cfg.hg_batched_chunk
        else:
            cp_Hg = compute_Hg

        H, g, order3 = cp_Hg(
            self.param_struct,
            loss,
            direction,
            **cp_kwargs,
        )

        return H.detach(), g.detach(), order3.detach()

    @increment_step
    def step(self):
        # Function that performs an update
        def make_step(direction):
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    for p in group["params"]:
                        p.add_(direction[i], alpha=-group["lr"])
                        i += 1

        # Evaluate the objective ONCE, at the current parameters, and reuse the graph
        # for both the gradient and the summaries.
        loss = self.loss_fn()
        self.logs["loss"].append(loss.detach())

        # Fill `.grad` so that the updater can produce a direction. `create_graph=True`
        # keeps the graph alive for compute_Hg below.
        upd_params = [p for gr in self.updater.param_groups for p in gr["params"]]
        grads = torch.autograd.grad(
            loss, upd_params, create_graph=True, allow_unused=True, materialize_grads=True
        )
        for p, gr in zip(upd_params, grads, strict=True):
            p.grad = gr.detach()

        # Compute the direction
        direction = self.param_struct.reindex(self.updater.compute_step(), self._dir_perm)

        # Normalize if required
        if self.cfg.normalize_dirs:
            self.dir_norm = self.normalize_dirs_(direction)

        # Compute H, g, order3 at the current point
        H, g, order3 = self.compute_Hg(loss, direction)

        # Store logs of H, g, order3
        self.logs["H"].append(H)
        self.logs["g"].append(g)
        self.logs["order3"].append(order3)

        # Compute order3_
        order3_ = order3.abs().pow(1 / 3)

        if self.cfg.noregul:
            # no regularization
            lrs = torch.linalg.solve(H, g)
        elif not self.cfg.nesterov.use:
            # with regularization, but no Nesterov cubic regul
            # => Tikhonov regularization
            regul_H = self.cfg.ridge * torch.eye(H.size(0), dtype=self.dtype, device=self.device)
            lrs = torch.linalg.solve(H + regul_H, g)
        else:
            # regularization with Nesterov cubic
            nest = self.cfg.nesterov
            lrs, lrs_logs = nesterov_lrs(
                H,
                g,
                order3_,
                damping_int=nest.damping_int,
                threshold_D_sing=nest.threshold_D_sing,
                hard_case_rtol=nest.hard_case_rtol,
                refine=nest.refine,
            )

            for k, v in lrs_logs.items():
                kk = "nesterov." + k
                if kk not in self.logs.keys():
                    self.logs[kk] = []
                self.logs[kk].append(v)

            if not lrs_logs["found"]:
                # `found=False` has two very different causes and the caller needs to
                # tell them apart. `infeasible_*` means nesterov_lrs did NOT fail: it
                # correctly reported that the cubic model is unbounded below, which
                # happens either on a genuinely unbounded objective or, spuriously, at a
                # stationary point of a quadratic -- there order3 = 0 exactly, so D = 0,
                # the whole space is ker(D), and Hbar = u' A u with u = grad f is ~1e-31
                # near the solution and rounds to something numerically indefinite.
                # Anything else is a solver failure.
                how = lrs_logs["x0.computation"]
                if how.startswith("infeasible"):
                    raise RuntimeError(
                        f"the cubic model is unbounded below ({how}): Hbar is not "
                        "positive definite on ker(D), so no step exists. If this happens "
                        f"at a stationary point (||gbar||_inf = {float(g.abs().max()):.3e}), "
                        "stop on a convergence criterion before calling step() again."
                    )
                raise RuntimeError(
                    f"nesterov_lrs failed to return a step ({how}); lrs not updated."
                )

        ## Additional operations on the lrs
        if self.cfg.remove_negative:
            lrs = lrs.relu()

        ## Assign lrs
        self.logs["lrs_clipped"].append(lrs)
        self.logs["curr_lrs"].append(lrs)
        for group, lr in zip(self.param_groups, lrs, strict=True):
            group["lr"] = group["damping"] * lr.item()

        # Store logs of lrs
        self.logs["lrs"].append(
            torch.tensor(
                [group["lr"] for group in self.param_groups], device=self.device, dtype=self.dtype
            )
        )

        ### To finish: perform update ###
        make_step(direction)
