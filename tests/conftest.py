import pytest
import torch

from grnewt import ParamStructure
from grnewt import partition as build_partition


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed():
    """Every test starts from the same RNG state. Autouse: no opt-in needed."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def f64():
    """Numerical tests need float64; restore the default afterwards."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield torch.float64
    torch.set_default_dtype(prev)


# --------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.fixture(params=_devices())
def device(request):
    return torch.device(request.param)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class MLP(torch.nn.Module):
    """Uniform-width MLP: reproduces the shape-aliasing conditions of BigMLP."""

    def __init__(self, widths, act=torch.tanh, bias=True):
        super().__init__()
        self.act = act
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(a, b, bias=bias) for a, b in zip(widths[:-1], widths[1:])
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class Quadratic(torch.nn.Module):
    """L(theta) = 0.5 theta^T A theta + b^T theta, with A and b known.

    Gives an oracle that does not go through autograd: Hbar = V A V^T exactly,
    and every third derivative is exactly zero.
    """

    def __init__(self, sizes, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n = sum(sizes)
        M = torch.randn(n, n, generator=g, dtype=torch.get_default_dtype())
        self.register_buffer("A", 0.5 * (M + M.T) + n * torch.eye(n))
        self.register_buffer("b", torch.randn(n, generator=g, dtype=torch.get_default_dtype()))
        self.blocks = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(s, generator=g, dtype=torch.get_default_dtype()))
            for s in sizes
        )

    def theta(self):
        return torch.cat([p.reshape(-1) for p in self.blocks])

    def forward(self, x):
        t = self.theta()
        return (0.5 * t @ self.A @ t + self.b @ t).expand(x.shape[0], 1)


@pytest.fixture
def mlp(f64, device):
    return MLP([6, 5, 5, 3]).to(device=device, dtype=f64)


@pytest.fixture
def uniform_mlp(f64, device):
    """All hidden weights share a shape: silent misalignment is possible here."""
    return MLP([4, 4, 4, 4]).to(device=device, dtype=f64)


@pytest.fixture
def batch(f64, device):
    def make(model, n=9):
        d_in = model.layers[0].in_features
        d_out = model.layers[-1].out_features
        return (
            torch.randn(n, d_in, device=device, dtype=f64),
            torch.randn(n, d_out, device=device, dtype=f64),
        )

    return make


@pytest.fixture
def mse():
    def full_loss_factory(model):
        return lambda x, y: (model(x) - y).pow(2).mean()

    return full_loss_factory


# --------------------------------------------------------------------------
# partitions
# --------------------------------------------------------------------------

PARTITION_BUILDERS = {
    "canonical": build_partition.canonical,
    "trivial": build_partition.trivial,
    "wb": build_partition.wb,
    "blocks-2": lambda m: build_partition.blocks(m, 2),
}


@pytest.fixture(params=sorted(PARTITION_BUILDERS))
def partition_name(request):
    return request.param


@pytest.fixture
def param_struct(mlp, partition_name):
    pgroups, _ = PARTITION_BUILDERS[partition_name](mlp)
    return ParamStructure(pgroups)


@pytest.fixture
def direction(param_struct):
    """A generic direction, ordered like param_struct.tup_params.

    Deliberately NOT all-ones: an all-ones direction is invariant under
    permutation and therefore cannot detect ordering bugs.
    """
    return tuple(torch.randn_like(p) for p in param_struct.tup_params)
