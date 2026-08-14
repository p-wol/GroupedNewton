import pytest
import torch
from numpy.polynomial.polynomial import Polynomial

import grnewt


def _apply_poly(poly, x):
    d = poly.degree()
    acc = 0
    pow_x = x.pow(0)

    for i in range(d + 1):
        acc = acc + poly.coef[i] * pow_x
        pow_x = pow_x * x

    return acc


@pytest.fixture
def poly_mono_1():
    return Polynomial([1, 2, -1, 0.5, -0.2, 0.1])


@pytest.fixture
def point_x_1():
    return torch.nn.Parameter(torch.ones(1).squeeze())


@pytest.fixture
def direction_1():
    return [torch.ones(1).squeeze()]


def test_diff_n_poly_mono(poly_mono_1, point_x_1, direction_1):
    poly = poly_mono_1
    x = point_x_1
    y = None
    direction = direction_1

    # Compute with grnewt
    param_struct = grnewt.ParamStructure([{"params": [x]}])
    order = poly.degree() + 1

    def full_loss(x_, y_):
        return _apply_poly(poly, x_)

    diff_grnewt = grnewt.diff_n(param_struct, order, full_loss, x, y, direction)
    diff_grnewt = [next(iter(dct.values())).item() for dct in diff_grnewt]
    diff_grnewt = torch.tensor(diff_grnewt)

    # Compute with numpy
    x_item = x.item()
    diff_np = [poly(x_item)]
    for m in range(1, order + 1):
        diff_np.append(poly.deriv(m)(x_item))
    diff_np = torch.tensor(diff_np).float()

    assert torch.allclose(diff_grnewt, diff_np)
