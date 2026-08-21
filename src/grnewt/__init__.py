"""
grnewt -- second-order optimization using a partition of the parameters.
 
Reference implementation for
  Wolinski, "Gathering and Exploiting Higher-Order Information when Training
  Large Structured Models", arXiv:2312.03885.
 
Public API::
 
    from grnewt import ParamStructure, compute_Hg, nesterov_lrs, NewtonSummary
    from grnewt import partition
 
The package depends only on torch, numpy and scipy. Models and datasets used
in the paper's experiments live in `experiments/`, not here.
"""

from . import optimizers, partition
from .differential import diff_n, diff_n_fullbatch
from .fullbatch import fullbatch_gradient
from .hg import compute_Hg, compute_Hg_batched, compute_Hg_fullbatch

#from .hg_batched import compute_Hg_batched
from .nesterov import nesterov_lrs
from .newton_stochastic_hv import NewtonStochasticHv
from .newton_summary import NewtonSummary
from .newton_summary_fb import NewtonSummaryFB
from .newton_summary_uniform_avg import NewtonSummaryUniformAvg
from .newton_summary_vanilla import NewtonSummaryVanilla
from .param_struct import ParamStructure
from .reduce_damping_on_plateau import ReduceDampingOnPlateau


# `datasets` needs torchvision and `models` pulls it in transitively; both belong
# to the `experiments` extra. Importing them eagerly made `import grnewt` fail on
# a `pip install -e ".[dev]"` environment -- i.e. on every CI `test` job -- which
# contradicts this module's own docstring. Deferring them keeps `grnewt.datasets`
# and `grnewt.models` working as attributes for anyone who has the extra.
def __getattr__(name):
    if name in ("datasets", "models", "loader_pre_hooks"):
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"datasets", "models", "loader_pre_hooks"})

__all__ = [
    # subpackages
    "datasets",
    "models",
    "optimizers",
    "partition",
    "loader_pre_hooks",
    # core
    "ParamStructure",
    "compute_Hg",
    "compute_Hg_fullbatch",
    "compute_Hg_batched",
    "nesterov_lrs",
    "fullbatch_gradient",
    "diff_n",
    "diff_n_fullbatch",
    # optimizers
    "NewtonSummary",
    "NewtonSummaryFB",
    "NewtonSummaryUniformAvg",
    "NewtonSummaryVanilla",
    "NewtonStochasticHv",
    "ReduceDampingOnPlateau",
    # misc
    "__version__",
]

