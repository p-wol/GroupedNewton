"""Typed schema for the `optimizer.hg` config node.

Single source of truth for every hyperparameter of the NewtonSummary family: name,
type, default, one-line description, and *which optimizers actually read it*.
Adding a hyperparameter costs one line here plus its use site.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

# Optimizer names, as used by `args.optimizer.name`.
NS = "NewtonSummary"
NSFB = "NewtonSummaryFB"
NSUA = "NewtonSummaryUniformAvg"
NSSA = "NewtonSummaryStaticAvg"
NSMA = "NewtonSummaryMovexpAvg"
ALL_NS = frozenset({NS, NSFB, NSUA, NSSA, NSMA})
STO_NS = frozenset({NSUA, NSSA, NSMA})


def P(help: str, used_by, **kw: Any) -> Any:
    """A dataclass field carrying its documentation and its consumers.

    `used_by` is the set of optimizer names whose code path actually reads the field.
    A field set to a non-default value by a run whose optimizer is not in `used_by`
    is a user error, reported by `check_consumed`.
    """
    if isinstance(used_by, str):
        used_by = {used_by}
    return field(metadata={"help": help, "used_by": frozenset(used_by)}, **kw)


class Partition(Enum):
    """Values accepted by `optimizer.hg.partition`.

    `blocks` and `alternate` take an integer argument, carried by `partition_arg`,
    instead of being spelled 'blocks-4' and parsed with str.find() at run time.
    """

    canonical = "canonical"
    trivial = "trivial"
    wb = "wb"
    blocks = "blocks"
    alternate = "alternate"
    vgg = "vgg"
    perceptron = "perceptron"


class UpdaterName(Enum):
    SGD = "SGD"
    Adam = "Adam"


@dataclass(kw_only=True, slots=True)
class DampingScheduleCfg:
    """Replaces the legacy string `damping_schedule: '<final>-<epoch>'` / `'None'`.

    That spelling forced training_hydra.py to str.split('-') and float()/int() the
    pieces at run time, i.e. a malformed value failed after the allocation started.
    """

    use: bool = P(
        "geometrically decay `damping` over the first `epoch` epochs", ALL_NS, default=False
    )
    final: float = P("target value of `damping` at epoch `epoch`", ALL_NS, default=1.0)
    epoch: int = P("epoch at which `damping` reaches `final`", ALL_NS, default=0)

    def __post_init__(self):
        if self.use and self.final <= 0:
            raise ValueError(f"damping_schedule.final must be > 0, got {self.final}")
        if self.use and self.epoch < 1:
            raise ValueError(f"damping_schedule.epoch must be >= 1, got {self.epoch}")


@dataclass(kw_only=True, slots=True)
class UpdaterCfg:
    name: UpdaterName = P(
        "inner optimizer producing the search direction u", ALL_NS, default=UpdaterName.SGD
    )
    momentum: float = P("momentum of the updater (SGD only)", STO_NS, default=0.9)
    momentum_damp: float = P("dampening of the updater (SGD only)", STO_NS, default=0.0)


@dataclass(kw_only=True, slots=True)
class NesterovCfg:
    use: bool = P(
        "solve the anisotropic cubic subproblem instead of H^{-1} g", ALL_NS, default=False
    )
    damping_int: float = P(
        "lambda_int >= 0; strength of the cubic regularization", ALL_NS, default=1.0
    )
    threshold_D_sing: float = P(
        "relative threshold below which d_i counts as zero; 0.0 is the only "
        "affine-invariant choice (nesterov.py, Appendix E)",
        ALL_NS,
        default=0.0,
    )
    hard_case_rtol: float = P(
        "relative tolerance defining the near-null eigenspace of K + c x0 I in the hard case",
        ALL_NS,
        default=1e-12,
    )
    refine: bool = P(
        "safeguarded Newton refinement of (r, eta); NOT validated over the fuzz set (STATE.md, N3)",
        ALL_NS,
        default=False,
    )

    def __post_init__(self):
        if self.damping_int < 0:
            raise ValueError(f"nesterov.damping_int must be >= 0, got {self.damping_int}")
        if not 0.0 <= self.threshold_D_sing < 1.0:
            raise ValueError(
                "nesterov.threshold_D_sing is a *relative* threshold and must "
                f"lie in [0, 1), got {self.threshold_D_sing}"
            )


@dataclass(kw_only=True, slots=True)
class UniformAvgCfg:
    period: int = P("Hg-update steps between two swaps of (X^a, X^b)", {NSUA}, default=1)
    warmup: int = P(
        "Hg-update steps during which H, g, D are averaged but the network is not trained",
        {NSUA},
        default=0,
    )

    def __post_init__(self):
        if self.period < 1:
            raise ValueError(f"uniform_avg.period must be >= 1, got {self.period}")
        if self.warmup < 0:
            raise ValueError(f"uniform_avg.warmup must be >= 0, got {self.warmup}")


@dataclass(kw_only=True, slots=True)
class StaticAvgCfg:
    nsamples: int = P("Number of samples to estimate E[H], E[g], E[order3]", {NSSA}, default=1)

    def __post_init__(self):
        if self.nsamples < 1:
            raise ValueError(f"static_avg.nsamples must be >= 1, got {self.nsamples}")


@dataclass(kw_only=True, slots=True)
class MovexpAvgCfg:
    movavg: float = P(
        "Exponential moving average update coefficient to estimage E[H], E[g], E[order3]",
        {NSMA},
        default=0.1,
    )

    def __post_init__(self):
        if self.movavg < 0 or self.movavg > 1:
            raise ValueError(f"movexp_avg.movavg must be in [0, 1], got {self.movavg}")


@dataclass(kw_only=True, slots=True)
class DmpAutoCfg:
    use: bool = P("reduce damping on plateau", ALL_NS, default=False)
    apply_to: str = P("attribute the scheduler acts on", ALL_NS, default="damping")
    patience: int = P("scheduler patience, in epochs", ALL_NS, default=1)
    cooldown: int = P("scheduler cooldown, in epochs", ALL_NS, default=0)
    threshold: float = P(
        "relative improvement below which a step counts as a plateau", ALL_NS, default=0.9
    )
    factor: float = P("multiplicative factor applied on plateau", ALL_NS, default=0.9)


@dataclass(kw_only=True, slots=True)
class HgCfg:
    """The whole `optimizer.hg` node."""

    # --- data used to estimate (gbar, Hbar, order3) ---------------------------------
    batch_size: int = P(
        "batch size for the (H, g) estimation; -1 = dataset batch size", ALL_NS, default=-1
    )
    partition: Partition = P("group construction rule", ALL_NS, default=Partition.canonical)
    partition_arg: int | None = P(
        "integer argument of partition in {blocks, alternate}; unused otherwise",
        ALL_NS,
        default=None,
    )
    partition_str: str | None = P(
        "string argument of partition in {vgg, perceptron}", ALL_NS, default=None
    )

    # --- the reduced model ----------------------------------------------------------
    diagonal: bool = P("compute only the diagonal of Hbar", ALL_NS, default=False)
    noregul: bool = P("bypass every regularization: lrs = Hbar^{-1} gbar", ALL_NS, default=False)
    ridge: float = P("ridge added to Hbar when nesterov.use is False", ALL_NS, default=0.0)

    # --- step size ------------------------------------------------------------------
    damping: float = P("per-group damping; multiplies the computed lr", ALL_NS, default=1.0)
    period_hg: int = P("training steps between two recomputations of (H, g)", STO_NS, default=1)
    normalize_dirs: bool = P(
        "normalize each proposition of 'direction' (on each subset of params)  before computing Hbar and gbar",
        ALL_NS,
        default=False,
    )
    mom_lrs: float = P("momentum on the learning rates", STO_NS | {NSFB}, default=0.0)
    maintain_true_lrs: bool = P(
        "keep the unclipped lrs as the momentum state", STO_NS | {NSFB}, default=True
    )
    remove_negative: bool = P("clamp negative learning rates to zero", ALL_NS, default=False)

    # --- compuation path ------------------------------------------------------------
    hg_batched: bool = P("use the batched version of compute_Hg", ALL_NS, default=False)
    hg_batched_chunk: int = P(
        "chunk_size in the batched version of compute_hg; -1 = S (partition size)",
        ALL_NS,
        default=-1,
    )

    # --- bookkeeping ----------------------------------------------------------------
    nologs: bool = P("do not dump the (H, g, lrs) logs", ALL_NS, default=False)

    # --- sub-nodes ------------------------------------------------------------------
    damping_schedule: DampingScheduleCfg = field(default_factory=DampingScheduleCfg)
    updater: UpdaterCfg = field(default_factory=UpdaterCfg)
    nesterov: NesterovCfg = field(default_factory=NesterovCfg)
    uniform_avg: UniformAvgCfg = field(default_factory=UniformAvgCfg)
    static_avg: StaticAvgCfg = field(default_factory=StaticAvgCfg)
    movexp_avg: MovexpAvgCfg = field(default_factory=MovexpAvgCfg)
    dmp_auto: DmpAutoCfg = field(default_factory=DmpAutoCfg)

    def __post_init__(self):
        if self.period_hg < 1:
            raise ValueError(f"period_hg must be >= 1, got {self.period_hg}")
        if self.damping <= 0:
            raise ValueError(f"damping must be > 0, got {self.damping}")
        if not 0.0 <= self.mom_lrs < 1.0:
            raise ValueError(f"mom_lrs must be in [0, 1), got {self.mom_lrs}")
        if self.ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")
        if self.partition in (Partition.blocks, Partition.alternate):
            if self.partition_arg is None:
                raise ValueError(f"partition={self.partition.value} requires partition_arg")
        elif self.partition_arg is not None:
            raise ValueError(f"partition_arg is meaningless for partition={self.partition.value}")
        if self.partition in (Partition.vgg, Partition.perceptron):
            if self.partition_str is None:
                raise ValueError(f"partition={self.partition.value} requires partition_str")
        elif self.partition_str is not None:
            raise ValueError(f"partition_str is meaningless for partition={self.partition.value}")
        if self.noregul and self.nesterov.use:
            raise ValueError(
                "noregul=True and nesterov.use=True are mutually exclusive: "
                "noregul short-circuits the cubic solver (newton_summary*.py)"
            )


@dataclass(kw_only=True, slots=True)
class LogsCfg(HgCfg):
    """The `logs_hg` node: run a NewtonSummaryFB in observation mode every epoch.

    It is an `HgCfg` plus `use`, deliberately: the point of the diagnostic is to report
    what a full-batch optimizer WOULD compute at this point, so it must be configurable
    exactly like one and must go through the same validation. A parallel schema would
    have to be kept in sync by hand with every field added to HgCfg, and `check_consumed`
    would not apply to it.

    Only the fields NewtonSummaryFB reads have an effect here; the rest are rejected by
    `check_consumed(cfg, "NewtonSummaryFB")` like anywhere else, so a dead setting in the
    logs node is an error rather than a silent no-op.
    """

    use: bool = P(
        "compute (Hbar, gbar, order3, lrs) in full batch at every epoch", ALL_NS, default=False
    )


# ---------------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------------


def _walk(cfg, prefix: str = ""):
    """Yield (dotted_path, value, field_object) over a nested dataclass instance."""
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        path = f"{prefix}{f.name}"
        if is_dataclass(v):
            yield from _walk(v, prefix=f"{path}.")
        else:
            yield path, v, f


def check_consumed(cfg: HgCfg, optimizer_name: str, *, prefix: str = "optimizer.hg") -> list[str]:
    """Fields set away from their default but not read by `optimizer_name`.

    Returns a list of human-readable diagnostics; empty means the config is fully
    honoured. This turns 'silently ignored' into an explicit, pre-submission error.

    `prefix` names the config node in the message: the same schema now backs both
    `optimizer.hg` and `logs_hg`.
    """
    default = type(cfg)()
    out: list[str] = []
    for path, value, f in _walk(cfg):
        used_by = f.metadata.get("used_by")
        if used_by is None or optimizer_name in used_by:
            continue
        dflt = getattr(_resolve(default, path.split(".")[:-1]), f.name)
        if value != dflt:
            out.append(
                f"{prefix}.{path} = {value!r} (default {dflt!r}) is not read by "
                f"{optimizer_name}; it is read by {sorted(used_by)}"
            )
    return out


def _resolve(obj, path_parts):
    for p in path_parts:
        obj = getattr(obj, p)
    return obj


def migrate(node):
    """Rewrite legacy spellings *before* the merge.

    Necessary, not optional: the merge rejects unknown keys and invalid enum values,
    so a legacy value can never be repaired inside `__post_init__` -- the config never
    gets there. Applied to the raw node, it keeps every existing sweep script and every
    `optimizer.hg.partition=blocks-4` command line working.
    """
    from omegaconf import OmegaConf

    node = OmegaConf.create(OmegaConf.to_container(node, resolve=True))
    ds = node.get("damping_schedule", None)
    if isinstance(ds, str):
        if ds == "None":
            node["damping_schedule"] = {"use": False}
        else:
            final, _, ep = ds.partition("-")
            node["damping_schedule"] = {"use": True, "final": float(final), "epoch": int(ep)}

    p = node.get("partition", None)
    if isinstance(p, str) and "-" in p:
        head, _, arg = p.partition("-")
        if head in ("blocks", "alternate"):
            node["partition"], node["partition_arg"] = head, int(arg)
        elif head in ("vgg", "perceptron"):
            node["partition"], node["partition_str"] = head, arg
    return node


def from_dictconfig(node, *, optimizer_name: str | None = None, strict: bool = True) -> HgCfg:
    """Validate a composed `optimizer.hg` DictConfig and return a real HgCfg.

    Call this once, at the boundary. Everything downstream sees a plain dataclass:
    typed, autocompleted by the IDE, and ~2 orders of magnitude faster to read than a
    DictConfig (measured: ~13 us vs ~26 ns per nested attribute access).
    """
    from omegaconf import OmegaConf

    cfg: HgCfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(HgCfg), node))
    if optimizer_name is not None:
        problems = check_consumed(cfg, optimizer_name)
        if problems and strict:
            raise ValueError(
                f"{len(problems)} setting(s) would be silently ignored by "
                f"{optimizer_name}:\n  " + "\n  ".join(problems)
            )
    return cfg


def from_dictconfig_logs_hg(node, *, strict: bool = True) -> LogsCfg:
    """Validate a composed `logs_hg` DictConfig and return a real LogsCfg.

    Same boundary discipline as `from_dictconfig`, and the same schema underneath, so
    `migrate()` applies unchanged and there is nothing to keep in sync. The diagnostic
    is a NewtonSummaryFB, so that is what the config is checked against.
    """
    from omegaconf import OmegaConf

    cfg: LogsCfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(LogsCfg), node))
    problems = check_consumed(cfg, NSFB, prefix="logs_hg")
    if problems and strict:
        raise ValueError(
            f"{len(problems)} setting(s) in the logs_hg node would be silently ignored "
            "by the NewtonSummaryFB used to compute them:\n  " + "\n  ".join(problems)
        )
    return cfg


def markdown_table() -> str:
    """Render the schema as documentation. Generated, therefore never stale."""
    lines = ["| field | type | default | read by | description |", "|---|---|---|---|---|"]
    for path, _, f in _walk(HgCfg()):
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        d = f.default if f.default is not dataclasses.MISSING else "-"
        d = d.value if isinstance(d, Enum) else d
        used = f.metadata.get("used_by", frozenset())
        used_s = "all" if used == ALL_NS else ", ".join(sorted(used))
        lines.append(f"| `{path}` | `{t}` | `{d!r}` | {used_s} | {f.metadata.get('help', '')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_table())
