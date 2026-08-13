import pytest
import torch
import grnewt
from grnewt import compute_Hg, compute_Hg_batched, ParamStructure
from grnewt import partition as build_partition
from conftest import f64


@pytest.mark.parametrize("build", [build_partition.canonical, build_partition.trivial, build_partition.wb])
def test_match_hg_vs_hg_batched(f64, build, chunk_size: int = 2):
    """Compare against grnewt.compute_Hg on a small MLP with a RANDOM direction."""

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 4), torch.nn.Tanh(),
        torch.nn.Linear(4, 3), torch.nn.Tanh(),
        torch.nn.Linear(3, 2),
    )
    x, yt = torch.randn(11, 5), torch.randn(11, 2)

    def full_loss(x_, y_):
        return (model(x_) - y_).pow(2).mean()

    pg, _ = build(model)
    ps = ParamStructure(pg)
    # NOTE: direction must be ordered like ps.tup_params, not model.parameters()
    direction = tuple(torch.randn_like(p) for p in ps.tup_params)

    H0, g0, o0 = compute_Hg(ps, full_loss, x, yt, direction)
    H1, g1, o1 = compute_Hg_batched(ps, full_loss, x, yt, direction, chunk_size=chunk_size)

    """
    print(f"{build.__name__:<10} S={ps.nb_groups}  "
          f"dH={(H0 - H1).abs().max():.2e}  "
          f"dg={(g0 - g1).abs().max():.2e}  "
          f"do3={(o0 - o1).abs().max():.2e}")
    """

    assert torch.allclose(H0, H1, atol=1e-9)
    assert torch.allclose(g0, g1, atol=1e-9)
    assert torch.allclose(o0, o1, atol=1e-9)
