import itertools
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .config import HgCfg
from .nesterov import nesterov_lrs
from .param_struct import ParamStructure


@dataclass(kw_only=True, slots=True)
class UpdateInstructions:
    recompute_lrs: bool
    do_update: bool


def increment_step(func):
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.step_counter += 1
        return ret

    return wrapper


class NSBase(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        full_loss,
        data_loader: DataLoader | None,
        updater,
        *,
        loader_pre_hook,
        cfg: HgCfg,
    ):
        """
        param_groups: param_groups of the model
        full_loss: full_loss(x, y) = l(m(x), y)
        data_loader: generates the data points used to estimate H and g
        cfg: validated `optimizer.hg` node; see grnewt/config.py for every field,
             its default, and which optimizers read it. Nothing else is read from
             the config, and every field this optimizer ignores is rejected at
             composition time by grnewt.config.check_consumed.
        """
        # `data_loader` is the source of the minibatches a subclass draws to ESTIMATE
        # the summaries. A subclass that computes them exactly (NewtonSummaryFB) draws
        # none and passes None: it must not be handed a loader, because a loader it is
        # not allowed to iterate is exactly what invites the re-entrancy bug.
        if data_loader is None:
            self.fn_data_loader = None
            self.dl_iter = None
        else:
            self.fn_data_loader = create_infinite_data_loader(data_loader)
            self.dl_iter = iter(self.fn_data_loader())
        self.full_loss = full_loss
        self.updater = updater
        self.loader_pre_hook = loader_pre_hook
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

    def compute_avg_Hg(self, direction):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # step() is split in three so that the diagnostic path (probe) can reuse the
    # arithmetic without reusing the side effects. Keeping them together and adding a
    # `dry_run` flag does not work: `dry_run` only suppressed the parameter update,
    # while group["lr"], curr_lrs, step_counter and self.logs were still mutated, so a
    # logger sharing this code path silently perturbed the run it was observing.
    # ------------------------------------------------------------------ #

    def _direction(self):
        """The candidate direction u, in ParamStructure order, normalized if asked."""
        direction = self.param_struct.reindex(self.updater.compute_step(), self._dir_perm)
        if self.cfg.normalize_dirs:
            self.dir_norm = self.normalize_dirs_(direction)
        return direction

    def _solve_lrs(self, H, g, order3):
        """Solve the reduced problem. Pure: returns (lrs, found, nesterov_logs)."""
        order3_ = order3.abs().pow(1 / 3)

        if self.cfg.noregul:
            return torch.linalg.solve(H, g), True, {}
        if not self.cfg.nesterov.use:
            regul_H = self.cfg.ridge * torch.eye(H.size(0), dtype=self.dtype, device=self.device)
            return torch.linalg.solve(H + regul_H, g), True, {}

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
        return lrs, bool(lrs_logs["found"]), lrs_logs

    def probe(self):
        """(Hbar, gbar, order3, lrs) at the current point, with NO side effect.

        Parameters, group["lr"], curr_lrs, step_counter and self.logs are all left
        untouched, so this can be called on a separate NewtonSummaryFB instance --
        possibly with a different partition and a different cfg -- to log what a
        full-batch step WOULD be, without perturbing the run.

        The one effect it cannot avoid: `updater.compute_step()` recomputes `.grad`
        (FBGDUpdate zeroes and refills it). Call it before the training step of the
        epoch, not between a `backward()` and a `step()`.

        Returns a plain dict, safe to `torch.save`: lrs is None when the reduced
        problem has no solution, and `nesterov.*` entries are present only when the
        cubic solver was used.
        """
        direction = self._direction()
        H, g, order3, _ = self.compute_avg_Hg(direction)
        lrs, found, nest_logs = self._solve_lrs(H, g, order3)
        out = {"H": H, "g": g, "order3": order3, "lrs": lrs if found else None}
        out.update({f"nesterov.{k}": v for k, v in nest_logs.items()})
        return out

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

        direction = self._direction()

        # Compute the averages of H, g, order3
        H, g, order3, update_instr = self.compute_avg_Hg(direction)

        # XXX: if all non-Hg updates normalize the direction, then, as the norm
        #      of direction decreases, our method will *overshoot* the objective
        #      for lrs. Unresolved issue.

        ### If we do not need to recompute the lrs ###
        if not update_instr.recompute_lrs:
            # Do immediately an update if necessary, then end step
            if update_instr.do_update:
                make_step(direction)
            return

        ### Now, we know that update_instr.recompute_lrs is True ###
        ### => compute the lrs                                   ###

        # Store logs of H, g, order3
        self.logs["H"].append(H)
        self.logs["g"].append(g)
        self.logs["order3"].append(order3)

        lrs, lrs_found, lrs_logs = self._solve_lrs(H, g, order3)
        for k, v in lrs_logs.items():
            self.logs.setdefault("nesterov." + k, []).append(v)
        if not lrs_found:
            print("Nesterov did not converge: lr not updated during this step.")
            lrs = self.curr_lrs

        ## Additional operations on the lrs
        r = self.cfg.mom_lrs if self.step_counter > 0 else 0
        self.curr_lrs = r * self.curr_lrs + (1 - r) * lrs
        lrs = self.curr_lrs
        if self.cfg.remove_negative:
            lrs = lrs.relu()

        ## Assign lrs
        self.logs["lrs_clipped"].append(lrs)
        self.logs["curr_lrs"].append(self.curr_lrs)
        for group, lr in zip(self.param_groups, lrs, strict=False):
            group["lr"] = group["damping"] * lr.item()

        # Store logs of lrs
        self.logs["lrs"].append(
            torch.tensor(
                [group["lr"] for group in self.param_groups], device=self.device, dtype=self.dtype
            )
        )

        ### To finish: perform update if necessary ###
        if update_instr.do_update:
            make_step(direction)


def create_infinite_data_loader(data_loader):
    # XXX: if the batch_size does not divide the total number of samples in
    #      the data_loader, then this may fail (possibly batches of irregular sizes)
    def f():
        for dl in itertools.repeat(data_loader):
            yield from dl

    return f
