from . import partition
from . import models
from . import datasets
from . import optimizers
from .util import fullbatch_gradient
from .hg import compute_Hg, compute_Hg_fullbatch
from .nesterov import nesterov_lrs
from .newton_summary import NewtonSummary
from .newton_summary_fb import NewtonSummaryFB
from .reduce_damping_on_plateau import ReduceDampingOnPlateau
from .differential import taylor_n, diff_1, pearlmutter_n, features_n
