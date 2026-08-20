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

from . import datasets, models, optimizers, partition
from .param_struct import ParamStructure
from .differential import diff_n, diff_n_fullbatch
from .hg import compute_Hg, compute_Hg_fullbatch
#from .hg_batched import compute_Hg_batched
from .nesterov import nesterov_lrs
from .newton_stochastic_hv import NewtonStochasticHv
from .newton_summary import NewtonSummary
from .newton_summary_fb import NewtonSummaryFB
from .newton_summary_uniform_avg import NewtonSummaryUniformAvg
from .newton_summary_vanilla import NewtonSummaryVanilla
from .reduce_damping_on_plateau import ReduceDampingOnPlateau
from .fullbatch import fullbatch_gradient

__all__ = [
    # subpackages
    "optimizers",
    "partition",
    "loader_pre_hooks",
    # core
    "ParamStructure",
    "compute_Hg",
    "compute_Hg_fullbatch",
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
 

